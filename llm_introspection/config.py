import os
import torch

class Config:
    """
    Configuration settings for the LLM Introspection framework.

    Attributes:
        llm_model_name: Name of the pre-trained LLM model.
        batch_size: Size of each batch for training.
        num_epochs: Number of epochs for training.
        learning_rate: Learning rate used in training.
        output_dir: Directory where output files will be saved.
        device: Compute device (e.g., 'cuda' or 'cpu').
    """
    def __init__(self, config_path=None):
        # Load from configuration file if available
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.llm_model_name: str = "decapoda-research/llama-2-7b"
            self.batch_size: int = 32
            self.num_epochs: int = 10
            self.learning_rate: float = 1e-4
            self.output_dir: str = "output/"
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def load_config(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)

    def save_config(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
