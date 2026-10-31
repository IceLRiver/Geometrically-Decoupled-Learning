"""Observation-only API for the released framework and recent baselines."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/v5'))
from inference import load_model,predict_states
