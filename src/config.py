import argparse
import yaml
from omegaconf import DictConfig

def load_config():
    parser = argparse.ArgumentParser(description="Run Bayesian Neural Network demo for geophysical parameters.")
    parser.add_argument("--config", type=str, default="configs/filter_tcn_base.yaml", help="Path to the configuration file.")
    parser.add_argument("--retrain", action="store_true", help="Force to ratrain the model.")
    parser.add_argument("--resample", action="store_true", help="Force to resample the data.")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = DictConfig(config)
    config.retrain = args.retrain
    config.resample = args.resample
    return config