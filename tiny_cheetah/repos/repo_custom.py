from __future__ import annotations

import os
import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, List

import requests

DEFAULT_FILES = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
}


class RepoCustom:
    """Custom downloader that avoids subprocess usage common in snapshot_download."""

    def __init__(
        self,
        model_name: str,
        cache_root: Path | None = None,
        backend: str | None = None,
    ) -> None:
        from tiny_cheetah.models.llm.backend import backend_model_config_class

        self.model_name = model_name
        self.backend = backend
        sanitized = model_name.replace("/", "__")
        self.base_dir = (cache_root or Path.home() / ".cache" / "tiny_cheetah_models") / sanitized
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.model_config = backend_model_config_class(backend=backend)()

    async def download(
        self,
        extra_files: Iterable[str] | None = None,
        revision: str = "main",
        progress_callback: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> tuple[Path, Any, List[str]]:
        messages: List[str] = []

        async def emit(message: str) -> None:
            messages.append(message)
            if progress_callback is None:
                return
            result = progress_callback(message)
            if inspect.isawaitable(result):
                await result

        has_cached_files = os.path.exists(self.base_dir) and any(self.base_dir.iterdir())
        if has_cached_files:
            self._load_configs()
            if not self._missing_cached_files(extra_files):
                return self.base_dir, self.model_config, messages

        repo_tree = await self._fetch_file_list(revision)
        repo_files = [entry["path"] for entry in repo_tree if entry.get("type") == "file"]

        wanted = set(DEFAULT_FILES)
        if extra_files:
            wanted.update(extra_files)
        safetensors = [path for path in repo_files if path.endswith(".safetensors")]
        wanted.update(safetensors)

        download_files = [file for file in repo_files if file in wanted]
        if not download_files:
            download_files = repo_files  # fallback, grab everything available
        if has_cached_files:
            download_files = [file for file in download_files if not (self.base_dir / file).exists()]
            if not download_files:
                self._load_configs()
                return self.base_dir, self.model_config.config, messages

        total_files = len(download_files)
        for index, filename in enumerate(download_files, start=1):
            await emit(f"Downloading {index}/{total_files}: {filename}")
            await self._download_file(filename, revision)
            await emit(f"Finished {index}/{total_files}: {filename}")

        self._load_configs()
        await emit("Download complete.")
        return self.base_dir, self.model_config.config, messages

    def _missing_cached_files(self, extra_files: Iterable[str] | None = None) -> list[str]:
        required = set(extra_files or [])
        if str(self.model_config.config.get("model_type", "")).lower() == "gpt_oss":
            required.add("chat_template.jinja")
        return sorted(str(filename) for filename in required if not (self.base_dir / filename).exists())

    def _load_configs(self) -> None:
        config_file = self.base_dir / "config.json"
        if config_file.exists():
            self.model_config.load(config_file)

        gen_config = self.base_dir / "generation_config.json"
        if gen_config.exists():
            self.model_config.load_generation_config(gen_config)

    async def _fetch_file_list(self, revision: str) -> list[dict]:
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"https://huggingface.co/api/models/{self.model_name}/tree/{revision}"
        response = await asyncio.to_thread(requests.get, url, headers=headers, params={"recursive": 1}, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch file list for {self.model_name}@{revision}: "
                f"{response.status_code} {response.text}"
            )
        tree = response.json()
        if not isinstance(tree, list):
            raise RuntimeError("Unexpected response when listing repository files.")
        return tree

    async def _download_file(self, filename: str, revision: str) -> None:
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        base_url = f"https://huggingface.co/{self.model_name}/resolve/{revision}/{filename}"
        response = await asyncio.to_thread(requests.get, base_url, headers=headers, stream=True, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {filename}: {response.status_code} {response.text}")
        destination = self.base_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
