# CrashX portable-project transfer

This Git repository contains the code, processed annotations and splits, saved
prediction/metric artifacts, audits, experiment configurations, and paper
documentation needed to inspect and reproduce CrashX.

It intentionally excludes local heavyweight artifacts that ordinary GitHub
cannot host:

| Artifact | Local path | Reason |
|---|---|---|
| CCD videos | `video1500/` | Dataset media; do not publish without confirming its license/redistribution terms. |
| QLoRA checkpoints | `outputs/` | Individual adapter files exceed GitHub's 100 MB Git limit. |
| Python environment | `.venv/` | Machine-specific and reproducible from requirements. |
| Smoke-test weights | `results/architecture_v4/smoke*/` | Large derived checkpoint artifacts. |

## Copy to the new server

Clone the Git repository, create a Python environment, and copy protected
media/checkpoints through an approved server-to-server path. Do not add the
videos or checkpoints to ordinary Git history.

```bash
git clone https://github.com/harshalDharpure/crash-x-vlm-gen.git
cd crash-x-vlm-gen
python3 -m venv .venv
.venv/bin/pip install -r crashx/requirements.txt

# Run from the destination server after replacing SOURCE_HOST.
rsync -aP SOURCE_HOST:/DATA/vaneet_2221cs15/vlm_new_gen/video1500/ video1500/
rsync -aP SOURCE_HOST:/DATA/vaneet_2221cs15/vlm_new_gen/outputs/ outputs/
```

The second command is optional for new experiments but needed to reproduce the
historical adapter inference. Model downloads from Hugging Face require the
appropriate account/license access.

## Integrity notes

- The original splits and historical predictions are retained for provenance.
- `crashx/data/splits_source_grouped_provisional/` and
  `crashx/data/splits_candidate_quarantined/` are diagnostic artifacts, not
  certified final benchmarks.
- Read `docs/source_grouping_protocol.md` before running new experiments.
