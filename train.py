#!/usr/bin/env python3
"""
Example training script for the Tiny Cheetah LLM stack.

This provides a minimal fine-tuning loop that reuses the existing model,
configuration, and weight-loading helpers. It is intentionally simple so it can
serve as a starting point for custom projects.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import tinygrad as tg
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from transformers import AutoTokenizer

from tiny_cheetah.models.llm.helpers import load_safetensors
from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.shard import Shard
from tiny_cheetah.models.llm.model_config import ModelConfig
from tiny_cheetah.repos import RepoHuggingFace


@dataclass
class Batch:
    """
    Simple container for input, target, and mask tensors.
    """
    input_ids: tg.Tensor
    labels: tg.Tensor
    attention_mask: tg.Tensor
    position_ids: tg.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Tiny Cheetah LLM.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model identifier to download (required unless --config-path is supplied)."
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default=None,
        help="Tokenizer identifier or path; required if --model-id is omitted."
    )
    parser.add_argument(
        "--tokenizer-file",
        type=Path,
        default=None,
        help="Optional explicit path (or repo/file) to a tokenizer.json/tokenizer.model."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to a config.json file for custom models (skips download)."
    )
    parser.add_argument(
        "--generation-config-path",
        type=Path,
        default=None,
        help="Optional generation_config.json path paired with --config-path."
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory containing model safetensors to load instead of downloading."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=False,
        default=None,
        help="Path to a UTF-8 text file used for fine-tuning. If omitted, provide --dataset-id."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Optional Hugging Face dataset identifier (e.g. NousResearch/Hermes-3-Dataset)."
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache downloaded/processed datasets."
    )
    parser.add_argument(
        "--max-dataset-entries",
        type=int,
        default=None,
        help="Optional cap on the number of dataset conversations to process."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only use locally cached Hugging Face files (no network)."
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=256,
        help="Training sequence length (tokens per sample)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of sequences per batch."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of complete passes over the dataset."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the Adam optimizer."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("TC_DEVICE", "CPU"),
        help="Tinygrad device string (e.g. CPU, CUDA, METAL)."
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Update weights every N steps instead of every step."
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to store safetensor checkpoints."
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Skip loading the pretrained safetensors and train from scratch."
    )
    return parser.parse_args()


def ensure_required_keys(config_obj) -> None:
    """
    Guarantee optional config entries exist so the model constructor does not
    raise KeyError. This handles gaps in raw HF config files.
    """
    if hasattr(config_obj, "model_config"):
        backing = config_obj.model_config
    else:
        backing = config_obj

    backing.setdefault("attn_scale", None)
    backing.setdefault("mlp_scale", None)
    backing.setdefault("temperature", None)
    backing.setdefault("top_k", None)
    backing.setdefault("top_p", None)


def parse_remote_identifier(identifier: Path, default_filename: str) -> tuple[str, str]:
    """
    Interpret a non-existent path as a Hugging Face repo identifier plus an optional filename.
    Returns (repo_id, filename).
    """
    raw = identifier.as_posix().strip()
    raw = raw.lstrip("./")
    if not raw or raw.startswith("/"):
        raise FileNotFoundError(f"Invalid remote identifier: {identifier}")

    if raw.endswith(".json"):
        repo_id, _, filename = raw.rpartition("/")
        if not repo_id:
            raise FileNotFoundError(
                f"Remote identifier '{raw}' must include repo id before the filename."
            )
        return repo_id, filename

    return raw, default_filename


def fetch_model_file(
    repo_id: str,
    filename: str,
    cache_dir: Optional[Path] = None,
    local_only: bool = False
) -> Path:
    """
    Download a single file from a Hugging Face model repo, falling back to RepoHuggingFace if needed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download remote config files. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    kwargs = {"repo_id": repo_id, "filename": filename, "repo_type": "model"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_only:
        kwargs["local_files_only"] = True

    try:
        local_path = hf_hub_download(**kwargs)
        return Path(local_path)
    except Exception as err:
        if local_only:
            raise FileNotFoundError(
                f"File '{filename}' not found in local cache for repo '{repo_id}'."
            ) from err
        print(
            f"[config] Direct download failed for {repo_id}/{filename}: {err}. "
            "Falling back to RepoHuggingFace snapshot download."
        )
        repo = RepoHuggingFace(repo_id)
        snapshot_path, _ = repo.download()
        candidate = snapshot_path / filename
        if candidate.exists():
            return candidate

        matches = list(snapshot_path.rglob(filename))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Unable to locate {filename} in repo {repo_id} after snapshot download."
        ) from err


def download_dataset(dataset_id: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Download a dataset from Hugging Face and return the local snapshot path.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download datasets. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    kwargs = {"repo_id": dataset_id, "repo_type": "dataset"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    path = snapshot_download(**kwargs)
    return Path(path)


def _conversation_to_text(record: dict) -> Optional[str]:
    candidates = record.get("conversations") or record.get("messages") or record.get("turns")
    if not candidates:
        return None

    turns: List[str] = []
    for turn in candidates:
        role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") or turn.get("content") or ""
        content = content.strip()
        if not content:
            continue
        prefix = f"{role.strip()}: " if role else ""
        turns.append(prefix + content)

    if not turns:
        return None
    return "\n".join(turns)


def iter_hermes_text(dataset_root: Path, limit: Optional[int] = None) -> Iterator[str]:
    """
    Yield conversation strings from the Hermes dataset snapshot.
    """
    jsonl_files = sorted(dataset_root.rglob("*.jsonl"))
    json_files = sorted(
        path for path in dataset_root.rglob("*.json")
        if path.name not in {"dataset_info.json", "info.json"}
    )

    emitted = 0

    def maybe_yield(text_value: Optional[str]) -> bool:
        nonlocal emitted
        if text_value:
            yield_again = limit is None or emitted < limit
            if yield_again:
                emitted += 1
                return True
        return False

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _conversation_to_text(record)
                if maybe_yield(text):
                    yield text
                    if limit is not None and emitted >= limit:
                        return

    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            maybe_data = data.get("data") or data.get("examples") or data.get("conversations")
            if isinstance(maybe_data, list):
                data = maybe_data
        if not isinstance(data, list):
            continue

        for record in data:
            if not isinstance(record, dict):
                continue
            text = _conversation_to_text(record)
            if maybe_yield(text):
                yield text
                if limit is not None and emitted >= limit:
                    return


def prepare_hermes_corpus(
    dataset_root: Path,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None
) -> Path:
    """
    Convert Hermes conversations into a newline-delimited text corpus and return the file path.
    """
    if output_dir is None:
        output_dir = dataset_root / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "hermes_corpus.txt"

    if corpus_path.exists():
        return corpus_path

    count = 0
    with corpus_path.open("w", encoding="utf-8") as out:
        for text in iter_hermes_text(dataset_root, limit=limit):
            out.write(text)
            out.write("\n\n")
            count += 1

    if count == 0:
        raise RuntimeError(
            f"No conversations found when processing Hermes dataset under {dataset_root}."
        )

    print(f"[dataset] Prepared {count} Hermes conversations at {corpus_path}")
    return corpus_path


def resolve_tokenizer_asset(
    tokenizer_id: str,
    cache_dir: Optional[Path],
    local_only: bool
) -> Optional[Path]:
    """Locate a tokenizer asset (json or model) for the given repo."""
    if not tokenizer_id or "/" not in tokenizer_id:
        return None

    candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.sp.model",
        "tokenizer.spm"
    ]

    for name in candidates:
        try:
            return fetch_model_file(tokenizer_id, name, cache_dir=cache_dir, local_only=local_only)
        except FileNotFoundError:
            continue
    return None


def tokenize_corpus(
    tokenizer: AutoTokenizer,
    data_path: Path,
    seq_length: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert the flat text corpus into aligned (input, target) sequences of
    length `seq_length`. Targets are the next-token shift of inputs.
    """
    print(f"[tokenizer] processing corpus at {data_path} with seq-length={seq_length}")
    text = data_path.read_text(encoding="utf-8")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False
    )
    ids = encoded["input_ids"]

    sequences: List[Tuple[np.ndarray, np.ndarray]] = []
    stride = seq_length
    window = seq_length + 1
    total_chunks = (len(ids) + stride - 1) // stride
    for start in range(0, len(ids) - window, stride):
        chunk = ids[start:start + window]
        print(f"Tokenizing chunk {len(sequences)+1}/{total_chunks}", end='\r')
        if len(chunk) < window:
            break
        inputs = np.asarray(chunk[:-1], dtype=np.int32)
        targets = np.asarray(chunk[1:], dtype=np.int32)
        sequences.append((inputs, targets))

    return sequences


def batches_from_sequences(
    sequences: Sequence[Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    device: str
) -> Iterable[Batch]:
    """
    Yield mini-batches constructed from the prepared token sequences.
    """
    for start in range(0, len(sequences), batch_size):
        slice_ = sequences[start:start + batch_size]
        if len(slice_) < batch_size:
            return

        inputs = np.stack([inp for inp, _ in slice_])
        targets = np.stack([tgt for _, tgt in slice_])
        attention_mask = np.ones_like(inputs, dtype=np.float32)
        position_ids = np.tile(
            np.arange(inputs.shape[1], dtype=np.int32),
            (inputs.shape[0], 1)
        )

        yield Batch(
            input_ids=tg.Tensor(inputs, device=device, dtype=tg.dtypes.int32),
            labels=tg.Tensor(targets, device=device, dtype=tg.dtypes.int32),
            attention_mask=tg.Tensor(attention_mask, device=device),
            position_ids=tg.Tensor(position_ids, device=device, dtype=tg.dtypes.int32)
        )


def save_checkpoint(model: Model, save_dir: Path, step: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    state = get_state_dict(model)
    checkpoint_path = save_dir / f"model_step_{step}.safetensors"
    safe_save(state, str(checkpoint_path))
    print(f"[checkpoint] saved to {checkpoint_path}")


def train_epoch(
    model: Model,
    optimizer: tg.optim.Optimizer,
    criterion: tg.nn.loss.CrossEntropyLoss,
    batches: Iterable[Batch],
    grad_accum: int
) -> float:
    """
    Run one training epoch and return the mean loss.
    """
    model_loss = 0.0
    step_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(batches, start=1):
        logits = model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids
        )
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = batch.labels.reshape(-1).cast(tg.dtypes.default_int)

        loss = criterion(logits, labels)
        loss.backward()
        step_loss += loss.item()

        if step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        steps += 1
        model_loss += loss.item()

        if step % 10 == 0:
            avg = step_loss / min(step, 10)
            print(f"[train] step={step} loss={avg:.4f}")
            step_loss = 0.0

    if steps == 0:
        return math.nan
    return model_loss / steps


def main() -> None:
    args = parse_args()

    if args.config_path is None and args.model_id is None:
        raise ValueError("Provide either --config-path for a custom model or --model-id to download from Hugging Face.")

    model_path: Optional[Path] = None
    remote_repo_hint: Optional[str] = None

    if args.config_path is not None:
        config_loader = ModelConfig()
        config_candidate = args.config_path
        if config_candidate.exists():
            config_loader.load(config_candidate)
        else:
            repo_id, filename = parse_remote_identifier(config_candidate, "config.json")
            config_file = fetch_model_file(repo_id, filename, local_only=args.offline)
            config_loader.load(config_file)
            remote_repo_hint = repo_id
        if args.generation_config_path is not None:
            gen_candidate = args.generation_config_path
            if gen_candidate.exists():
                config_loader.load_generation_config(gen_candidate)
            else:
                repo_id, filename = parse_remote_identifier(
                    gen_candidate,
                    "generation_config.json"
                )
                gen_file = fetch_model_file(repo_id, filename, local_only=args.offline)
                config_loader.load_generation_config(gen_file)
                if remote_repo_hint is None:
                    remote_repo_hint = repo_id
        model_path = args.weights_dir
        if model_path is None and not args.from_scratch:
            print("[warn] No weights directory supplied; switching to --from-scratch.")
            args.from_scratch = True
    elif args.model_id is not None:
        repo = RepoHuggingFace(args.model_id)
        model_path, config_loader = repo.download()
        if args.weights_dir is not None:
            model_path = args.weights_dir
    else:
        # This branch should be unreachable because of the earlier guard.
        raise ValueError("Invalid configuration: missing model information.")

    ensure_required_keys(config_loader)
    config_dict = config_loader.model_config if hasattr(config_loader, "model_config") else config_loader

    model_name = args.model_id or remote_repo_hint or config_dict.get("model_type") or "custom"
    shard = Shard(
        model_name,
        start_layer=0,
        end_layer=config_dict["num_layers"],
        total_layers=config_dict["num_layers"] + 1
    )

    model = Model(
        config_loader,
        shard,
        use_tied=config_dict.get("tie_word_embeddings", False)
    )

    if not args.from_scratch:
        if model_path is None:
            raise ValueError("Weights directory must be provided when not training from scratch.")
        print("Loading pretrained weights...")
        load_safetensors(
            model,
            model_path,
            config_dict,
            weight_device=args.device,
            use_tied=config_dict.get("tie_word_embeddings", False)
        )

    tokenizer_file_path: Optional[Path] = None
    if args.tokenizer_file is not None:
        tokenizer_candidate = args.tokenizer_file
        if tokenizer_candidate.exists():
            tokenizer_file_path = tokenizer_candidate
        else:
            repo_id, filename = parse_remote_identifier(tokenizer_candidate, "tokenizer.json")
            tokenizer_file_path = fetch_model_file(repo_id, filename, local_only=args.offline)
            if remote_repo_hint is None:
                remote_repo_hint = repo_id

    tokenizer_id = args.tokenizer_id or args.model_id or remote_repo_hint
    tokenizer_asset: Optional[Path] = None
    if tokenizer_file_path is not None:
        tokenizer_asset = tokenizer_file_path
    elif isinstance(tokenizer_id, str) and "/" in tokenizer_id:
        try:
            tokenizer_asset = resolve_tokenizer_asset(tokenizer_id, None, args.offline)
        except FileNotFoundError:
            tokenizer_asset = None

    if tokenizer_id is None and tokenizer_asset is not None:
        tokenizer_id = str(tokenizer_asset.parent)

    if tokenizer_id is None:
        raise ValueError("Tokenizer identifier is required when --model-id is omitted. Provide --tokenizer-id.")

    tokenizer_kwargs: dict[str, object] = {"local_files_only": args.offline}
    if tokenizer_asset is not None:
        suffix = tokenizer_asset.suffix.lower()
        path_str = str(tokenizer_asset)
        if suffix == ".json":
            tokenizer_kwargs["tokenizer_file"] = path_str
            tokenizer_kwargs.setdefault("use_fast", True)
            tokenizer_kwargs.setdefault("legacy", False)
        else:
            tokenizer_kwargs["vocab_file"] = path_str
            tokenizer_kwargs.setdefault("legacy", True)

    print(f"[tokenizer] using '{tokenizer_id}'")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            **tokenizer_kwargs
        )
    except TypeError as exc:
        if tokenizer_asset is None:
            raise RuntimeError(
                "Failed to load tokenizer. The selected repository may not contain tokenizer assets."
                " Provide --tokenizer-file or point --tokenizer-id at a repo with tokenizer resources."
            ) from exc
        raise

    data_path = args.data_path
    if data_path is None:
        if args.dataset_id is None:
            raise ValueError("Provide --data-path or specify --dataset-id to download a dataset.")
        dataset_cache_root = args.dataset_cache_dir
        dataset_snapshot = download_dataset(args.dataset_id, dataset_cache_root)
        default_processed_root = Path.cwd() / "datasets" / args.dataset_id.replace("/", "__")
        processed_dir = (
            (dataset_cache_root / "processed") if dataset_cache_root is not None else default_processed_root
        )
        data_path = prepare_hermes_corpus(
            dataset_snapshot,
            processed_dir,
            limit=args.max_dataset_entries
        )
        print(f"[dataset] Using processed corpus at {data_path}")

    sequences = tokenize_corpus(tokenizer, data_path, args.seq_length)
    if not sequences:
        raise RuntimeError("Dataset is empty after tokenization; check seq-length or data.")

    optimizer = tg.optim.Adam(get_parameters(model), lr=args.lr)
    criterion = tg.nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        batches = batches_from_sequences(sequences, args.batch_size, args.device)
        avg_loss = train_epoch(model, optimizer, criterion, batches, args.gradient_accumulation)
        print(f"[epoch] {epoch} mean loss = {avg_loss:.4f}")

        if args.save_dir is not None:
            global_step += 1
            save_checkpoint(model, args.save_dir, global_step)


if __name__ == "__main__":
    main()
