from setuptools import setup, find_packages

setup(
    name="tiny-cheetah",
    version="0.1",
    description="Distributed inference and training using tinygrad",
    packages=find_packages(),
    install_requires=[
        "tinygrad",
        "numpy",
        "textual",
        "textual-dev"
    ],
)