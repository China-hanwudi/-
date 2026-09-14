"""Compatibility-free CH-SIMS_v2 entry point; delegates to the audited trainer."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_sims_v7 import main
if __name__=='__main__': main()
