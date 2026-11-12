import torch
from transformers import AutoProcessor
from crashx.models.train_qlora import CrashSFTJsonlDataset, build_collate_fn
p=AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',local_files_only=True)
ds=CrashSFTJsonlDataset('crashx/data/splits/train.jsonl')
b=build_collate_fn(p,16,336,8192)([ds[0],ds[1]])
for i in range(2):
 answer=p.tokenizer.decode(b['labels'][i][b['labels'][i]!=-100],skip_special_tokens=True).strip()
 assert answer==ds[i]['assistant_text'], (answer,ds[i]['assistant_text'])
 assert (b['labels'][i][b['attention_mask'][i]==0]==-100).all()
print('PASS: full assistant answers retained; prompt/video/padding masked', b['input_ids'].shape)
try:
 build_collate_fn(p,16,336,1)([ds[0]])
except ValueError as e:
 assert 'rather than truncating' in str(e)
 print('PASS: unsafe multimodal truncation rejected')
else:
 raise AssertionError('Unsafe truncation accepted')
