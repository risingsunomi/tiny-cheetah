from setuptools import setup, find_packages

BASE_REQUIRES = [
    "tinygrad",
    "numpy",
    "requests",
    "safetensors",
    "textual",
    "transformers",
    "huggingface-hub",
    "python-dotenv",
    "tokenizers",
    "zstandard",
    "torch",
]

EXTRAS = {
    "dev": [
        "pytest",
        "textual-dev",
    ],
}

setup(
    name="tiny_cheetah",
    version="0.1",
    description="Distributed inference and training with tinygrad or torch backends",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "tiny_cheetah.tui": ["*.tcss"],
        "tiny_cheetah.tui.widget": ["*.tcss"],
    },
    python_requires=">=3.10",
    install_requires=BASE_REQUIRES,
    extras_require=EXTRAS,
)
