"""Evaluate a frozen CMU-MOSEI checkpoint on valid.pt only."""
from __future__ import annotations
import argparse, torch
from .mosei_dataset import MOSEIPackedDataset
from .regression_config import N3RegressionConfig
from .regression_model import N3SentimentModel
from .train_mosei import evaluate
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--data',required=True); ap.add_argument('--batch-size',type=int,default=64); a=ap.parse_args(); c=torch.load(a.checkpoint,map_location='cpu',weights_only=True); cfg=N3RegressionConfig(**c['cfg']); m=N3SentimentModel(cfg); m.load_state_dict(c['model']); print(evaluate(m,MOSEIPackedDataset(a.data),torch.device('cpu'),a.batch_size))
if __name__=='__main__': main()
