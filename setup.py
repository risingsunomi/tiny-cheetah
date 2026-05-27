from __future__ import annotations

import os
import re


_BASE_TORCH_VERSION = "2.10.0"
_BASE_TORCH_ROCM_VERSION = "7.1"


def _torch_version() -> str:
    return os.getenv("TC_TORCH_VERSION", _BASE_TORCH_VERSION).strip()


def _torch_rocm_version() -> str:
    return os.getenv("TC_TORCH_ROCM_VERSION", _BASE_TORCH_ROCM_VERSION).strip()


def _normalize_cuda_tag(raw_variant: str) -> str | None:
    variant = raw_variant.strip().lower()
    if variant in {"", "auto", "cpu", "mps", "default"}:
        return None

    # Accept common formats:
    # cu126, cu12.6, cuda12.6, cuda-12.8, 12.6, 130
    direct = re.fullmatch(r"cu(\d{3,4})", variant)
    if direct:
        return f"cu{direct.group(1)}"

    dotted = re.fullmatch(r"(?:cu|cuda)?[-_ ]?(\d{1,2})[._](\d{1,2})", variant)
    if dotted:
        major = int(dotted.group(1))
        minor = int(dotted.group(2))
        return f"cu{major}{minor}"

    plain = re.fullmatch(r"(\d{3,4})", variant)
    if plain:
        return f"cu{plain.group(1)}"

    return None


def _normalize_rocm_tag(raw_variant: str) -> str | None:
    variant = raw_variant.strip().lower()
    if variant in {"rocm", "amd", "hip"}:
        version = _torch_rocm_version()
    else:
        match = re.fullmatch(r"(?:rocm|amd|hip)[-_ ]?(.+)", variant)
        if not match:
            return None
        version = match.group(1)

    version = version.strip().lower().replace("_", ".")
    compact = re.fullmatch(r"(\d)(\d)(\d?)", version)
    if compact and "." not in version:
        parts = [compact.group(1), compact.group(2)]
        if compact.group(3):
            parts.append(compact.group(3))
        return "rocm" + ".".join(parts)

    dotted = re.fullmatch(r"(\d+)(?:[.](\d+))?(?:[.](\d+))?", version)
    if dotted:
        parts = [part for part in dotted.groups() if part is not None]
        return "rocm" + ".".join(parts)

    return None


def _normalize_torch_variant_tag(raw_variant: str) -> str | None:
    rocm_tag = _normalize_rocm_tag(raw_variant)
    if rocm_tag is not None:
        return rocm_tag
    return _normalize_cuda_tag(raw_variant)


def _torch_requirement() -> str:
    """
    Resolve torch dependency from environment.

    Supported env vars:
    - TC_TORCH_VARIANT: auto|cpu|mps|cu126|cuda12.8|rocm|rocm7.1|hip6.4|...
    - TC_TORCH_VERSION: base torch version (default: 2.10.0)
    - TC_TORCH_ROCM_VERSION: version used by rocm/amd/hip aliases (default: 7.1)

    Notes:
    - CUDA/ROCm local-version wheels (e.g. +cu126, +rocm7.1) generally require
      the matching PyTorch wheel index to be configured in pip.
    """
    variant = os.getenv("TC_TORCH_VARIANT", "auto").strip().lower()
    version = _torch_version()

    # Default resolver path (PyPI / platform default wheel).
    tag = _normalize_torch_variant_tag(variant)
    if tag is None:
        return f"torch>={version}"
    return f"torch=={version}+{tag}"


BASE_REQUIRES = [
    "tinygrad",
    "numpy",
    "requests",
    "safetensors",
    "textual",
    "transformers",
    "jinja2",
    "huggingface-hub",
    "python-dotenv",
    "tokenizers",
    "zstandard",
    _torch_requirement(),
]

_DEFAULT_TORCH_VERSION = _torch_version()
_DEFAULT_TORCH_ROCM_TAG = _normalize_rocm_tag("rocm") or f"rocm{_BASE_TORCH_ROCM_VERSION}"

EXTRAS = {
    "dev": [
        "pytest",
        "textual-dev",
    ],
    # Use with pip wheel index URLs if selecting accelerator variants.
    "torch-cpu": [f"torch>={_DEFAULT_TORCH_VERSION}"],
    "torch-cu126": [f"torch=={_DEFAULT_TORCH_VERSION}+cu126"],
    "torch-cu128": [f"torch=={_DEFAULT_TORCH_VERSION}+cu128"],
    "torch-cu130": [f"torch=={_DEFAULT_TORCH_VERSION}+cu130"],
    "torch-rocm": [f"torch=={_DEFAULT_TORCH_VERSION}+{_DEFAULT_TORCH_ROCM_TAG}"],
}


def run_setup() -> None:
    from setuptools import find_packages, setup

    setup(
        name="cheetah",
        version="0.1",
        description="Distributed inference and training with tinygrad or torch backends",
        packages=find_packages(include=["cheetah", "cheetah.*"]),
        include_package_data=True,
        package_data={
            "cheetah.agent": ["functions.json"],
            "cheetah.agent.prompts": ["*.j2"],
            "cheetah.tui": ["*.tcss"],
            "cheetah.tui.widget": ["*.tcss"],
        },
        python_requires=">=3.10",
        install_requires=BASE_REQUIRES,
        extras_require=EXTRAS,
    )


if __name__ == "__main__":
    run_setup()
