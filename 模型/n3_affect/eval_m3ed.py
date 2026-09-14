"""Evaluate M3ED valid only; test evaluation is deliberately a separate frozen step."""
from __future__ import annotations
import argparse, json, torch
from .m3ed_dataset import M3EDPackedDataset
from .config import N3TrainConfig
from .model import N3EmotionModel
from .train_m3ed import evaluate
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--data',required=True); ap.add_argument('--batch-size',type=int,default=64); args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True); cfg=N3TrainConfig(**{k:(tuple(v) if k=='emotion_label_order' else v) for k,v in ck['cfg'].items()}); m=N3EmotionModel(cfg); m.load_state_dict(ck['model']); print(json.dumps(evaluate(m,M3EDPackedDataset(args.data),torch.device('cpu'),args.batch_size),indent=2))
if __name__=='__main__': main()
