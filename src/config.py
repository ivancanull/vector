import argparse
import yaml

try:
    from omegaconf import DictConfig
except ModuleNotFoundError:
    class DictConfig(dict):
        """Small attribute-access fallback used when OmegaConf is unavailable."""

        def __init__(self, values):
            super().__init__()
            for key, value in values.items():
                self[key] = self._wrap(value)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = self._wrap(value)

        @classmethod
        def _wrap(cls, value):
            if isinstance(value, dict):
                return cls(value)
            if isinstance(value, list):
                return [cls._wrap(item) for item in value]
            return value

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
