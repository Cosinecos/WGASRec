# WGASRec
**Multi Subband Sequential Recommendation via Wavelet Gated Attention and Structure Aware Graph Propagation**

## Acknowledgments
We thank the anonymous reviewers for their constructive and insightful feedback. We are grateful to the GroupLens team for MovieLens-1M, to the McAuley Lab (UC San Diego) for the Amazon Review Data (2018) and category files (e.g., All_Beauty), and to the developers of PyTorch and pytorch-wavelets. This repository builds upon open-source software; any remaining errors are our own.

## Overview
WGASRec performs subband-aware modeling with stationary wavelet transforms, lightweight per-band attention, energy/flatness-driven gating, and structure-aware graph propagation. It captures long-term trends and short-term bursts while remaining efficient and interpretable.

## Environment
- Python 3.10
- PyTorch ≥ 2.3
- pytorch-wavelets == 1.3.0
- numpy, pandas, PyYAML, tqdm, scikit-learn, matplotlib

```bash
pip install -r requirements.txt
```

## Repository Layout
```
wgasrec/
  utils/        registry, metrics, seed, graph ops
  data/         dataset loading (JSONL + meta.json)
  models/       WGASRec
  configs/      YAML configs (ml1m, beauty, sports, lastfm)
  train.py      training with early stopping (NDCG@10)
  eval.py       evaluation
```

## Data Format
For each dataset, produce:
- `train.jsonl`, `val.jsonl`, `test.jsonl`: each line has `{"user": int, "seq": [int, ...], "target": int}`
- `meta.json`: `{"num_items": int, "num_users": int}`

Sequences are sorted by timestamp; the last item per user is the target (leave-one-out). Item IDs start from 1 and 0 is reserved for padding.

## Datasets

### MovieLens-1M (ML-1M)
- Landing page: https://grouplens.org/datasets/movielens/1m/
- Direct zip: https://files.grouplens.org/datasets/movielens/ml-1m.zip

### Amazon-Beauty (All Beauty)
Official 2018 portal and direct category files from the McAuley Lab (UC San Diego).

- 2018 portal (documentation & links): https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Per-category “All Beauty” (reviews, JSON Lines, compressed): https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz
- Per-category “All Beauty” metadata (optional): https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz

Example:
```bash
mkdir -p data/beauty && cd data/beauty
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz
gunzip -k All_Beauty.json.gz
```

### Amazon-Sports (Sports and Outdoors)
- Portal: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Reviews (JSON Lines, compressed): https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Sports_and_Outdoors.json.gz

### LastFM (HetRec 2011)
- Official page: https://grouplens.org/datasets/hetrec-2011/
- Direct index: https://files.grouplens.org/datasets/hetrec2011/
  - Download `hetrec2011-lastfm-2k.zip` and follow the README for license/format.

## Preprocessing (Sketch)
1. Parse raw interactions (timestamp ascending) to build user sequences.
2. Map item ids to consecutive integers in `[1..num_items]`.
3. Leave-one-out: last item → `target`, the rest → `seq`.
4. Write JSONL files and `meta.json`.

## Training & Evaluation
```bash
python -m wgasrec.train --config wgasrec/configs/ml1m.yaml --data_dir /path/to/ml1m --out runs/ml1m
python -m wgasrec.eval  --config wgasrec/configs/ml1m.yaml --data_dir /path/to/ml1m --ckpt runs/ml1m/best.pt
```

## Citation
If you use the Amazon 2018 data, please cite:
- Jianmo Ni, Jiacheng Li, Julian McAuley. *Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects*. arXiv:1905.12742 (Amazon Review Data 2018).

If you use MovieLens or HetRec data, please follow the respective dataset licenses and citation instructions on their official pages.
