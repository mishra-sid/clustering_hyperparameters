"""
Scripts to train and evaluate model models with hyperparameter optimization
"""
import fastentrypoints
from setuptools import find_packages, setup

setup(
    name="clustering_hyperparameters",
    version="0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    description="Scripts to train and evaluate model models with hyperparameter optimization",
    install_requires=[
        "numpy",
        "xopen",
        "toml",
        "absl-py",
    ],
    extras_require={
        "test": ["pytest"],
    },
    entry_points={"console_scripts": ["clustering_hyperparameters = clustering_hyperparameters.__main__:main"]},
)
