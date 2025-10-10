#!/usr/bin/env python3
"""
Example training script for the Tiny Cheetah LLM stack.

This provides a minimal fine-tuning loop that reuses the existing model,
configuration, and weight-loading helpers. It is intentionally simple so it can
serve as a starting point for custom projects.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

SPIN_LOOP = ["|", "/", "-", "\\"]

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
    attention_mask: Optional[tg.Tensor]
    position_ids: tg.Tensor


def cross_entropy_loss(logits: tg.Tensor, targets: tg.Tensor) -> tg.Tensor:
    """Compute mean cross-entropy loss for logits and integer targets."""
    log_probs = logits.log_softmax(axis=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (-gathered).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Tiny Cheetah LLM.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model identifier to download (required unless --config-path is supplied)."
    )
    parser.add_argument(
        "--custom-model-id",
        type=str,
        default=None,
        help="Custom model identifier to use locally."
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


def extract_dataset(dataset_root: Path, limit: Optional[int] = None) -> Iterator[str]:
    """
    Yield conversation strings from a dataset
    """

    # check for jsonl files
    jsonl_files = sorted(dataset_root.rglob("*.jsonl"))
    json_files = sorted(
        path for path in dataset_root.rglob("*.json")
        if path.name not in {"dataset_info.json", "info.json"}
    )
    use_jsonl = True if jsonl_files else False

    # check for zst files
    zst_files = sorted(dataset_root.rglob("*.zst"))
    use_zst = True if not zst_files else False

    emitted = 0
    def maybe_yield(text_value: Optional[str]) -> bool:
        nonlocal emitted
        if text_value:
            yield_again = limit is None or emitted < limit
            if yield_again:
                emitted += 1
                return True
        return False

    if use_jsonl:
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
    elif use_zst:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "zstandard is required to read .zst dataset files. "
                "Install it with `pip install zstandard`."
            ) from exc

        dctx = zstd.ZstdDecompressor()
        for path in zst_files:
            with path.open("rb") as compressed:
                with dctx.stream_reader(compressed) as reader:
                    text_stream = tg.utils.BufferedLineReader(reader)
                    for line in text_stream:
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
    else:
        raise RuntimeError(f"No supported dataset (jsonl, .zst) files found under {dataset_root}.")

def prepare_dataset_corpus(
    dataset_root: Path,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None
) -> Path:
    """
    Converts dataset conversations into a newline-delimited text corpus and return the file path.
    """
    if output_dir is None:
        output_dir = dataset_root / "processed_" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "dataset_corpus.txt"

    if corpus_path.exists():
        return corpus_path

    count = 0
    with corpus_path.open("w", encoding="utf-8") as out:
        for text in extract_dataset(dataset_root, limit=limit):
            out.write(text)
            out.write("\n\n")
            count += 1

    if count == 0:
        raise RuntimeError(
            f"No conversations found when processing dataset under {dataset_root}."
        )

    print(f"[dataset] Prepared {count} dataset conversations at {corpus_path}")
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


def stream_corpus_batches(
    tokenizer: AutoTokenizer,
    data_path: Path,
    seq_length: int,
    batch_size: int,
    device: str,
    max_sequences: Optional[int] = None
) -> Iterator[Batch]:
    """Yield mini-batches by streaming tokens directly from the corpus file."""
    print(f"[data] tokenizing corpus from {data_path} with seq_length={seq_length}, batch_size={batch_size}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Corpus file {data_path} not found")

    token_buffer: List[int] = []
    seq_inputs: List[List[int]] = []
    seq_targets: List[List[int]] = []
    total_sequences = 0

    def flush_batch() -> Batch:
        nonlocal seq_inputs, seq_targets
        input_arr = np.asarray(seq_inputs, dtype=np.int32)
        target_arr = np.asarray(seq_targets, dtype=np.int32)
        position_arr = np.tile(
            np.arange(seq_length, dtype=np.int32),
            (input_arr.shape[0], 1)
        )
        batch = Batch(
            input_ids=tg.Tensor(input_arr, device=device, dtype=tg.dtypes.int32),
            labels=tg.Tensor(target_arr, device=device, dtype=tg.dtypes.int32),
            attention_mask=None,
            position_ids=tg.Tensor(position_arr, device=device, dtype=tg.dtypes.int32)
        )
        seq_inputs, seq_targets = [], []
        return batch

    with data_path.open("r", encoding="utf-8") as corpus:
        for line in corpus:
            tokens = tokenizer.encode(line, add_special_tokens=False)
            if not tokens:
                continue
            token_buffer.extend(tokens)

            while len(token_buffer) >= seq_length + 1:
                window = token_buffer[:seq_length + 1]
                seq_inputs.append(window[:-1])
                seq_targets.append(window[1:])
                total_sequences += 1
                del token_buffer[:seq_length]

                if len(seq_inputs) == batch_size:
                    yield flush_batch()

                if max_sequences is not None and total_sequences >= max_sequences:
                    break

            if max_sequences is not None and total_sequences >= max_sequences:
                break
       
    if seq_inputs:
        yield flush_batch()


def save_checkpoint(model: Model, save_dir: Path, step: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    state = get_state_dict(model)
    checkpoint_path = save_dir / f"model_step_{step}.safetensors"
    safe_save(state, str(checkpoint_path))
    print(f"[checkpoint] saved to {checkpoint_path}")

s
def train_epoch(
    model: Model,
    optimizer: tg.nn.optim.Optimizer,
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
    orig_training_flag = tg.Tensor.training
    tg.Tensor.training = True
    start_time = time.time()
    last_display = start_time
    total_tokens = 0

    try:
        for step, batch in enumerate(batches, start=1):
            logits = model(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids
            )
            vocab_size = logits.shape[-1]
            logits = logits.reshape(-1, vocab_size)
            labels = batch.labels.reshape(-1).cast(tg.dtypes.default_int)

            loss = cross_entropy_loss(logits, labels)
            loss.backward()
            step_loss += loss.item()

            seq_tokens = batch.input_ids.shape[0] * batch.input_ids.shape[1]
            total_tokens += seq_tokens
            now = time.time()
            if now - last_display >= 0.5:
                spinner = SPIN_LOOP[step % len(SPIN_LOOP)]
                elapsed = max(now - start_time, 1e-6)
                tok_rate = total_tokens / elapsed
                sys.stdout.write(
                    f"\r[train] {spinner} step={step} loss={loss.item():.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}"
                )
                sys.stdout.flush()
                last_display = now

            if step % grad_accum == 0:
                missing_grads = [p for p in optimizer.params if p.grad is None]
                for tensor in missing_grads:
                    tensor.grad = tg.Tensor.zeros_like(tensor)
                optimizer.step()
                optimizer.zero_grad()

            steps += 1
            model_loss += loss.item()

            if step % 10 == 0:
                avg = step_loss / min(step, 10)
                elapsed = max(time.time() - start_time, 1e-6)
                tok_rate = total_tokens / elapsed
                print(
                    f"\r[train] ✓ step={step} loss={avg:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}           "
                )
                step_loss = 0.0

        if steps == 0:
            return math.nan

        # Handle leftover gradients when grad_accum does not divide steps
        if steps % grad_accum != 0:
            missing_grads = [p for p in optimizer.params if p.grad is None]
            for tensor in missing_grads:
                tensor.grad = tg.Tensor.zeros_like(tensor)
            optimizer.step()
            optimizer.zero_grad()

        # final display
        elapsed = max(time.time() - start_time, 1e-6)
        tok_rate = total_tokens / elapsed
        print(
            f"\r[train] done steps={steps} mean_loss={model_loss / steps:.4f} total_tok={total_tokens} tok/s={tok_rate:.1f}          "
        )
        return model_loss / steps
    finally:
        tg.Tensor.training = orig_training_flag


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

    if args.custom_model_id is not None:
        model_name = args.custom_model_id
    else:
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
        data_path = prepare_dataset_corpus(
            dataset_snapshot,
            processed_dir,
            limit=args.max_dataset_entries
        )
        print(f"[dataset] Using processed corpus at {data_path}")

    parameters = [param.requires_grad_(True) for param in get_parameters(model)]
    optimizer = tg.nn.optim.Adam(parameters, lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        batches = stream_corpus_batches(
            tokenizer=tokenizer,
            data_path=data_path,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            device=args.device
        )
        avg_loss = train_epoch(model, optimizer, batches, args.gradient_accumulation)
        if math.isnan(avg_loss):
            raise RuntimeError(
                "No training batches were produced. Check dataset size, sequence length, or batch size."
            )
        print(f"[epoch] {epoch} mean loss = {avg_loss:.4f}")

        if args.save_dir is not None:
            global_step += 1
            save_checkpoint(model, args.save_dir, global_step)


if __name__ == "__main__":
    main()
