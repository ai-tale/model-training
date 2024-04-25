#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aitale-model-training",
    version="0.1.0",
    author="AI Tale Team",
    author_email="info@aitale.tech",
    description="Training pipelines and fine-tuning scripts for AI Tale's specialized story generation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-tale/model-training",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "datasets>=2.0.0",
        "tensorboard>=2.8.0",
        "wandb>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0"
    ],
    entry_points={
        "console_scripts": [
            "aitale-train=src.train.finetune:main",
            "aitale-prepare-data=src.data.prepare_dataset:main",
            "aitale-evaluate=src.evaluate.evaluate_model:main",
        ],
    },
) 