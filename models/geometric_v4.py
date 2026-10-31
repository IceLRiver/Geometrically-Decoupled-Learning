"""Public observation-only inference interface for the geometrically decoupled model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/v4'))
from train_models import Model as GeometricallyDecoupledModel
from neural_predict import load_checkpoint,predict
