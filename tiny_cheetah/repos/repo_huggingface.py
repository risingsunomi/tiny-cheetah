"""
HuggingFace repo for downloading models
"""
from glob import glob
from pathlib import Path

import huggingface_hub

from tiny_cheetah.models.llm.model_config import ModelConfig

class RepoHuggingFace:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = ModelConfig()

    def download(self) -> tuple[Path, ModelConfig]:
        # TO DO: for now download whole model, later make it only download safetensors
        # and other model information it needs
        # use tqdm to display on textual frontend
        model_data = huggingface_hub.snapshot_download(repo_id=self.model_name)

        for file in glob(f"{model_data}/**/*", recursive=True):
            path = Path(file)
            if path.name == "config.json":
                print(f"Loading config from {file}")
                self.model_config.load(file)

        return Path(model_data), self.model_config