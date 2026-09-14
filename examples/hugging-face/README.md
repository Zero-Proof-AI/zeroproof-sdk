# Hugging Face, both directions

Push a graded dataset to a Hub repo you own, pull any Hub split onto your
account and read its numbers before you train on it, and push a finished
training run's adapter as a model repo. One script, three calls. You need
`ZEROPROOF_API_KEY` and a Hugging Face account connected on the platform
(the import half of the script works on a public repo without the
connection); a minute end to end.

| call | what moves | where it lands |
|---|---|---|
| `zps.hf_publish("ds_...")` | your rows | `huggingface.co/datasets/<you>/<repo>`, one split per purpose, commit tagged `zp-<dataset id>` |
| `zps.import_hf("ns/name", split=...)` | any Hub split | a dataset on your account, profiled (pass rate, support, mixed prompts) |
| `zps.hf_publish_run("run_...")` | a run's LoRA adapter | `huggingface.co/<you>/<repo>` with a model card, private by default |

Versions are commits. Every push is tagged with the ZeroProof id it came
from, so `load_dataset(repo, split, revision="zp-ds_...")` loads exactly
that push, and `zeroproof.json` in the repo maps each split to its dataset
with history. Pushing a new cut into the same split replaces the old parts
and the commit message carries the delta.

## Run it

Connect your Hugging Face account once, on any dataset page at
https://www.zeroproofai.com/platform/datasets (the platform holds the
token, the SDK never sees it). Then:

```bash
pip install zeroproof
zeroproof login                                   # or ZEROPROOF_API_KEY
cd examples/hugging-face
python roundtrip.py                               # import a public split, profile it, print the numbers
python roundtrip.py --push ds_0123 --repo my-set  # also push one of your sets and print the tag
```

Output for the import half:

```
imported cornell-movie-review-data/rotten_tomatoes:test -> ds_... (1066 rows)
  rows 1066 · prompts 1 · graded 0 · pass – · support –
  (no reward field: profile it after grading, or use it as an eval set)
```

The imported set is deleted at the end unless you pass `--keep`.
