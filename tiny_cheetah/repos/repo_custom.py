from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import requests

from tiny_cheetah.models.llm.model_config import ModelConfig


DEFAULT_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
}


class RepoCustom:
    """Custom downloader that avoids subprocess usage common in snapshot_download."""

    def __init__(self, model_name: str, cache_root: Path | None = None) -> None:
        self.model_name = model_name
        sanitized = model_name.replace("/", "__")
        self.base_dir = (cache_root or Path.home() / ".cache" / "tiny_cheetah_models") / sanitized
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.model_config = ModelConfig()

    def download(self, extra_files: Iterable[str] | None = None, revision: str = "main") -> tuple[Path, ModelConfig, List[str]]:
        messages: List[str] = []

        repo_tree = self._fetch_file_list(revision)
        repo_files = [entry["path"] for entry in repo_tree if entry.get("type") == "file"]

        wanted = set(DEFAULT_FILES)
        if extra_files:
            wanted.update(extra_files)
        safetensors = [path for path in repo_files if path.endswith(".safetensors")]
        wanted.update(safetensors)

        download_files = [file for file in repo_files if file in wanted]
        if not download_files:
            download_files = repo_files  # fallback, grab everything available

        for filename in download_files:
            messages.append(f"Downloading {filename}")
            self._download_file(filename, revision)

        self._load_configs()
        return self.base_dir, self.model_config, messages

    def _load_configs(self) -> None:
        config_file = self.base_dir / "config.json"
        if config_file.exists():
            self.model_config.load(config_file)

        gen_config = self.base_dir / "generation_config.json"
        if gen_config.exists():
            self.model_config.load_generation_config(gen_config)

    def _fetch_file_list(self, revision: str) -> list[dict]:
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"https://huggingface.co/api/models/{self.model_name}/tree/{revision}"
        response = requests.get(url, headers=headers, params={"recursive": 1}, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch file list for {self.model_name}@{revision}: "
                f"{response.status_code} {response.text}"
            )
        tree = response.json()
        if not isinstance(tree, list):
            raise RuntimeError("Unexpected response when listing repository files.")
        return tree

    def _download_file(self, filename: str, revision: str) -> None:
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        base_url = f"https://huggingface.co/{self.model_name}/resolve/{revision}/{filename}"
        response = requests.get(base_url, headers=headers, stream=True, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {filename}: {response.status_code} {response.text}")
        destination = self.base_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
