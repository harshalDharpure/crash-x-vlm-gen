#!/usr/bin/env python3
"""4-bit QLoRA SFT for Qwen2.5-VL-7B-Instruct on CCD JSONL (CrashLogic-7B).

Saves adapters to outputs/crashlogic_7b_lora/ by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_qlora")

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class CrashSFTJsonlDataset(Dataset):
    """Minimal SFT dataset: video path + user prompt + assistant target."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self.rows: list[dict[str, Any]] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        user = r["messages"][0]["content"]
        assistant = r["messages"][1]["content"]
        return {
            "video_path": r["video_path"],
            "user_text": user,
            "assistant_text": assistant,
            "video_id": r["video_id"],
        }


def build_collate_fn(processor, num_frames: int = 8, max_side: int = 224, max_seq_len: int = 768):
    """Collate that samples keyframes, downsamples, and builds Qwen VL batches."""
    from crashx.data.dataset import load_video_frames_cv2
    from crashx.models.vru_baseline import frames_to_pil
    from crashx.models.video_preprocess import call_video_processor, resize_frames_np

    def collate(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        videos = []
        for ex in examples:
            frames = load_video_frames_cv2(ex["video_path"], num_frames=num_frames)
            if max_side > 0:
                frames = resize_frames_np(frames, max_side=max_side)
            pil_frames = frames_to_pil(frames)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames},
                        {"type": "text", "text": ex["user_text"]},
                    ],
                },
                {"role": "assistant", "content": ex["assistant_text"]},
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(prompt)
            videos.append(pil_frames)

        try:
            batch = call_video_processor(processor, texts, videos)
        except (TypeError, ValueError):
            batch = processor(
                text=texts,
                images=videos[0] if len(videos) == 1 else videos,
                padding=True,
                return_tensors="pt",
            )

        if max_seq_len > 0 and batch["input_ids"].shape[-1] > max_seq_len:
            raise ValueError(
                f"Sequence has {batch['input_ids'].shape[-1]} tokens, exceeding {max_seq_len}; "
                "increase --max-seq-len rather than truncating visual or answer tokens."
            )

        labels = batch["input_ids"].clone()
        # Supervise only assistant content, retaining the assistant EOS token.
        marker = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        for i in range(labels.shape[0]):
            ids = batch["input_ids"][i].tolist()
            starts = [j + len(marker) for j in range(len(ids) - len(marker) + 1)
                      if ids[j:j + len(marker)] == marker]
            if len(starts) != 1:
                raise ValueError("Expected exactly one assistant boundary in SFT example")
            labels[i, :starts[0]] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        if not (labels != -100).any(dim=1).all():
            raise ValueError("Example has no supervised answer tokens")
        batch["labels"] = labels
        return batch

    return collate


def build_training_args(
    output_dir: Path,
    args: argparse.Namespace,
    train_size: int,
    has_eval: bool,
) -> "TrainingArguments":
    """Build TrainingArguments compatible with the installed transformers version."""
    from transformers import TrainingArguments

    steps_per_epoch = max(
        1,
        (train_size + args.batch_size * args.grad_accum - 1)
        // (args.batch_size * args.grad_accum),
    )
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * 0.03))

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,
        "seed": args.seed,
        "data_seed": args.seed,
        "max_steps": args.max_steps,
        "bf16": torch.cuda.is_available(),
        "remove_unused_columns": False,
        "report_to": "none",
        "dataloader_num_workers": args.num_workers,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
    }

    sig = __import__("inspect").signature(TrainingArguments.__init__)
    if "warmup_ratio" in sig.parameters:
        kwargs["warmup_ratio"] = 0.03
    else:
        kwargs["warmup_steps"] = warmup_steps

    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch" if has_eval else "no"
    elif "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch" if has_eval else "no"

    return TrainingArguments(**kwargs)


def train(args: argparse.Namespace) -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        TrainingArguments,
        Trainer,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading processor/model: %s", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Full names prevent suffix matches from silently adapting vision MLPs.
    targets = [name for name, module in model.named_modules()
               if ".visual." not in name and name.rsplit(".", 1)[-1] in LORA_TARGETS
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("Could not locate language adapter targets")
    language_target_count = len(targets)
    if args.visual_lora:
        targets.extend([name for name, module in model.named_modules()
                        if ".visual.merger.mlp." in name and isinstance(module, torch.nn.Linear)])
        if len(targets) == language_target_count:
            raise RuntimeError("Could not locate visual merger linear layers")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = CrashSFTJsonlDataset(args.train_jsonl)
    eval_ds = CrashSFTJsonlDataset(args.val_jsonl) if args.val_jsonl else None
    collate = build_collate_fn(
        processor,
        num_frames=args.num_frames,
        max_side=args.max_side,
        max_seq_len=args.max_seq_len,
    )

    training_args = build_training_args(
        output_dir,
        args,
        train_size=len(train_ds),
        has_eval=eval_ds is not None,
    )

    # Raw multimodal rows must reach our collator without TRL text preprocessing.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )

    logger.info(
        "Starting QLoRA SFT: epochs=%s, train=%s, val=%s → %s",
        args.epochs,
        len(train_ds),
        len(eval_ds) if eval_ds else 0,
        output_dir,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    processor.save_pretrained(str(output_dir))
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    logger.info("Saved CrashLogic-7B LoRA adapters to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA SFT for CrashLogic-7B")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/crashlogic_7b_lora"))
    p.add_argument("--visual-lora", action="store_true", help="Adapt visual-to-language merger with LoRA")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-side", type=int, default=224, help="Max longer video frame side in pixels")
    p.add_argument("--max-seq-len", type=int, default=768, help="Truncate tokenized training sequences")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42, help="Global training seed for reproducibility")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — QLoRA 7B training will likely fail or be extremely slow.")
    train(args)


if __name__ == "__main__":
    main(sys.argv[1:])
