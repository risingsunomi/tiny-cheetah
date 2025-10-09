#!/usr/bin/env python3
"""
Minimal RLHF-style training loop for the Tiny Cheetah LLM stack.

This script illustrates how you could wire a policy model, optional reference
model, a simple reward function, and KL-regularised policy optimisation using
tinygrad. It is intentionally lightweight so you can adapt it to your own RLHF
experiments.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tinygrad as tg
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from transformers import AutoTokenizer

from tiny_cheetah.models.llm.helpers import load_safetensors, generate
from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.model_config import ModelConfig
from tiny_cheetah.models.llm.shard import Shard
from tiny_cheetah.repos import RepoHuggingFace


RewardFn = Callable[[str, str], float]


@dataclass
class PromptRecord:
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KL-regularised RLHF fine-tuning example.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face identifier for the policy model (required unless --config-path is provided)."
    )
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default=None,
        help="Tokenizer identifier or path; required if --model-id is omitted."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Use a local config.json instead of downloading."
    )
    parser.add_argument(
        "--generation-config-path",
        type=Path,
        default=None,
        help="Optional generation_config.json when using --config-path."
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory containing policy weights (safetensors)."
    )
    parser.add_argument(
        "--ref-weights-dir",
        type=Path,
        default=None,
        help="Directory containing frozen reference model weights."
    )
    parser.add_argument(
        "--prompts-path",
        type=Path,
        required=True,
        help="Text or JSONL file of prompts used for rollouts."
    )
    parser.add_argument(
        "--reward-script",
        type=Path,
        default=None,
        help="Path to a Python file exposing `score(prompt, response) -> float`."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("TC_DEVICE", "CPU"),
        help="Tinygrad device string (e.g. CPU, CUDA, METAL)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over the prompt list."
    )
    parser.add_argument(
        "--rollouts-per-epoch",
        type=int,
        default=32,
        help="How many prompt completions to sample per epoch."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per prompt."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for rollouts."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k filtering during generation (0 disables)."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p filtering during generation."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Policy optimiser learning rate."
    )
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=0.1,
        help="KL penalty weight against the reference model."
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory for checkpoint safetensors."
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Checkpoint every N optimisation steps."
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Skip loading weights and start from random parameters."
    )
    return parser.parse_args()


def ensure_required_keys(config_obj) -> None:
    if hasattr(config_obj, "model_config"):
        backing = config_obj.model_config
    else:
        backing = config_obj
    backing.setdefault("attn_scale", None)
    backing.setdefault("mlp_scale", None)
    backing.setdefault("temperature", None)
    backing.setdefault("top_k", None)
    backing.setdefault("top_p", None)


def load_reward_function(path: Optional[Path]) -> RewardFn:
    if path is None:
        def default_reward(prompt: str, response: str) -> float:
            length_bonus = min(len(response.strip().split()), 128) / 128.0
            diversity_penalty = response.lower().count(prompt.lower()) * 0.1
            return float(length_bonus - diversity_penalty)
        return default_reward

    if not path.exists():
        raise FileNotFoundError(f"Reward script {path} not found.")

    module = runpy.run_path(str(path))
    score_fn = module.get("score")
    if not callable(score_fn):
        raise ValueError(f"Reward script {path} must define a callable `score(prompt, response)`.")

    def reward(prompt: str, response: str) -> float:
        value = score_fn(prompt, response)
        return float(value)

    return reward


def load_prompts(path: Path) -> List[PromptRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file {path} not found.")

    prompts: List[PromptRecord] = []
    if path.suffix.lower() in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get("prompt") or data.get("instruction") or data.get("input")
                if text:
                    prompts.append(PromptRecord(prompt=text.strip()))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompts.append(PromptRecord(prompt=line.strip()))

    if not prompts:
        raise RuntimeError("No prompts were loaded; ensure the file is non-empty.")

    return prompts


def save_checkpoint(model: Model, save_dir: Path, step: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    state = get_state_dict(model)
    checkpoint_path = save_dir / f"policy_step_{step}.safetensors"
    safe_save(state, str(checkpoint_path))
    print(f"[checkpoint] saved to {checkpoint_path}")


def prepare_model(
    config_loader: ModelConfig,
    weights_dir: Optional[Path],
    device: str,
    trainable: bool,
    load_weights: bool,
    model_name: str
) -> Model:
    ensure_required_keys(config_loader)
    config_dict = config_loader.model_config

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

    if load_weights and weights_dir is not None:
        load_safetensors(
            model,
            weights_dir,
            config_dict,
            weight_device=device,
            use_tied=config_dict.get("tie_word_embeddings", False)
        )
    elif load_weights and weights_dir is None and trainable:
        raise ValueError("Weights directory must be provided when loading pretrained parameters.")

    if not trainable:
        for param in get_parameters(model):
            param.requires_grad = False

    return model


def build_inputs_with_response(
    prompt_ids: np.ndarray,
    response_tokens: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    response_array = np.asarray(response_tokens, dtype=np.int32)[None, :]
    full_ids = np.concatenate([prompt_ids, response_array], axis=1).astype(np.int32)
    attention = np.ones_like(full_ids, dtype=np.float32)
    position = (np.cumsum(attention, axis=1) - 1) * attention
    return full_ids, attention, position


def sequence_log_probs(
    model: Model,
    input_ids: tg.Tensor,
    attention_mask: tg.Tensor,
    position_ids: tg.Tensor,
    response_length: int
) -> Tuple[tg.Tensor, tg.Tensor]:
    logits = model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    response_logits = logits[:, -response_length:, :]
    targets = input_ids[:, -response_length:].cast(tg.dtypes.int32)
    log_probs = response_logits.log_softmax()
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    seq_log_prob = gathered.sum(axis=1)
    return seq_log_prob, gathered


def main() -> None:
    args = parse_args()

    if args.config_path is None and args.model_id is None:
        raise ValueError("Provide either --config-path for a custom policy or --model-id to download from Hugging Face.")

    policy_weights_dir: Optional[Path] = None

    if args.config_path is not None:
        if not args.config_path.exists():
            raise FileNotFoundError(f"Config path {args.config_path} does not exist.")
        policy_config = ModelConfig()
        policy_config.load(args.config_path)
        if args.generation_config_path is not None:
            if not args.generation_config_path.exists():
                raise FileNotFoundError(
                    f"Generation config path {args.generation_config_path} does not exist."
                )
            policy_config.load_generation_config(args.generation_config_path)
        policy_weights_dir = args.weights_dir
        if policy_weights_dir is None and not args.from_scratch:
            print("[warn] No policy weights supplied; toggling --from-scratch.")
            args.from_scratch = True
    elif args.model_id is not None:
        repo = RepoHuggingFace(args.model_id)
        download_dir, policy_config = repo.download()
        policy_weights_dir = args.weights_dir or download_dir
    else:
        # Guard should prevent reaching this.
        raise ValueError("Invalid policy configuration.")

    tokenizer_id = args.tokenizer_id or args.model_id
    if tokenizer_id is None:
        raise ValueError("Tokenizer identifier is required when --model-id is omitted. Provide --tokenizer-id.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
    prompts = load_prompts(args.prompts_path)
    reward_fn = load_reward_function(args.reward_script)

    model_name = args.model_id or policy_config["model_type"] or "custom"
    policy_model = prepare_model(
        policy_config,
        policy_weights_dir,
        args.device,
        trainable=True,
        load_weights=not args.from_scratch,
        model_name=model_name
    )

    reference_model: Optional[Model] = None
    if args.ref_weights_dir is not None:
        reference_model = prepare_model(
            policy_config,
            args.ref_weights_dir,
            args.device,
            trainable=False,
            load_weights=True,
            model_name=model_name
        )

    optimiser = tg.optim.Adam(get_parameters(policy_model), lr=args.learning_rate)

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== RLHF Epoch {epoch}/{args.epochs} ===")
        for rollout_idx in range(args.rollouts_per_epoch):
            prompt = prompts[rollout_idx % len(prompts)].prompt

            encoded = tokenizer(prompt, return_tensors="np")
            prompt_ids = encoded["input_ids"].astype(np.int32)
            attention_mask = encoded["attention_mask"]
            input_ids = tg.Tensor(prompt_ids, device=args.device, dtype=tg.dtypes.int32)
            attn_mask = tg.Tensor(attention_mask, device=args.device)
            position_ids = ((attn_mask.cumsum(axis=1) - 1) * attn_mask).cast(tg.dtypes.int32)

            generated_tokens = generate(
                model=policy_model,
                input_ids=input_ids,
                attention_mask=attn_mask,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temp=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                verbose=False
            )
            if not generated_tokens:
                print("[warn] Generation returned no tokens; skipping rollout.")
                continue

            response_length = len(generated_tokens)
            full_ids_np, full_attention_np, full_position_np = build_inputs_with_response(
                prompt_ids,
                generated_tokens
            )

            policy_input = tg.Tensor(full_ids_np, device=args.device, dtype=tg.dtypes.int32)
            policy_attention = tg.Tensor(full_attention_np, device=args.device)
            policy_positions = tg.Tensor(full_position_np, device=args.device, dtype=tg.dtypes.int32)

            seq_log_prob, token_log_probs = sequence_log_probs(
                policy_model,
                policy_input,
                policy_attention,
                policy_positions,
                response_length
            )

            ref_token_log_probs = None
            kl_term = tg.Tensor([0.0], device=args.device)
            if reference_model is not None:
                ref_input = tg.Tensor(full_ids_np, device=args.device, dtype=tg.dtypes.int32)
                ref_attention = tg.Tensor(full_attention_np, device=args.device)
                ref_position = tg.Tensor(full_position_np, device=args.device, dtype=tg.dtypes.int32)
                _, ref_token_log_probs = sequence_log_probs(
                    reference_model,
                    ref_input,
                    ref_attention,
                    ref_position,
                    response_length
                )
                kl_term = (token_log_probs - ref_token_log_probs).mean()

            response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            reward_value = reward_fn(prompt, response_text)
            reward_tensor = tg.Tensor([reward_value], device=args.device)

            policy_objective = (reward_tensor * seq_log_prob).mean()
            loss = -policy_objective + args.kl_beta * kl_term

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_steps += 1
            loss_value = float(loss.item())
            print(
                f"[rollout] step={total_steps} prompt_len={prompt_ids.shape[1]} "
                f"resp_len={response_length} reward={reward_value:.4f} loss={loss_value:.4f}"
            )
            print(f"[response] {response_text}\n")

            if args.save_dir is not None and total_steps % args.save_interval == 0:
                save_checkpoint(policy_model, args.save_dir, total_steps)

    if args.save_dir is not None:
        save_checkpoint(policy_model, args.save_dir, total_steps or 1)


if __name__ == "__main__":
    main()
