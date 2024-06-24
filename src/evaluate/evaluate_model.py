#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for story generation models.
"""

import argparse
import json
import logging
import os
import sys
import yaml
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel, 
    PreTrainedTokenizer
)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_metrics

# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned language model for story generation")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model to evaluate",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Path to the test dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file with generation parameters",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to JSON file with prompts for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer from path."""
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    
    return model, tokenizer


def load_test_dataset(test_dataset_path: str):
    """Load test dataset."""
    logger.info(f"Loading test dataset from {test_dataset_path}")
    
    extension = test_dataset_path.split(".")[-1]
    dataset = load_dataset(
        extension,
        data_files={"test": test_dataset_path},
        cache_dir=None,
    )["test"]
    
    return dataset


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from a JSON file."""
    with open(prompts_file, "r") as f:
        prompts = json.load(f)
    
    if isinstance(prompts, dict) and "prompts" in prompts:
        prompts = prompts["prompts"]
    
    if not isinstance(prompts, list):
        raise ValueError("Prompts file should contain a list of prompts or a dict with a 'prompts' key")
    
    return prompts


def generate_sample_texts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    prompts: Optional[List[str]] = None,
    num_samples: int = 3,
    **generation_kwargs
) -> List[str]:
    """Generate sample texts using the model."""
    if prompts is None:
        # Default story starters if no prompts provided
        prompts = [
            "Once upon a time in a magical forest,",
            "In a kingdom far, far away, there lived a young prince who",
            "The old wizard looked at the ancient book and whispered,",
            "The little girl found a mysterious door in her garden that",
            "Two friends were walking in the woods when suddenly",
        ]
    
    # Select a subset of prompts if num_samples is less than available prompts
    if num_samples < len(prompts):
        prompts = prompts[:num_samples]
    
    # Add more prompts by repeating if necessary
    while len(prompts) < num_samples:
        prompts.append(prompts[len(prompts) % len(prompts)])
    
    model.eval()
    generated_texts = []
    
    for prompt in prompts[:num_samples]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs
            )
        
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts


def evaluate_model_on_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 4,
    text_column: str = "text",
    **generation_kwargs
) -> Dict[str, float]:
    """Evaluate model on a dataset of stories."""
    model.eval()
    
    # Determine text column
    if text_column not in dataset.column_names:
        text_column = dataset.column_names[0]
        logger.warning(f"Text column '{text_column}' not found in dataset. Using '{text_column}' instead.")
    
    all_refs = []
    all_hyps = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        texts = batch[text_column]
        
        # For each text, use the first 50 tokens as prompt and predict the rest
        prompts = []
        references = []
        
        for text in texts:
            # Tokenize the text
            tokens = tokenizer.encode(text)
            # Use first 50 tokens (or less if shorter) as prompt
            prompt_length = min(50, len(tokens) // 3)  # Use first third, up to 50 tokens
            prompt_tokens = tokens[:prompt_length]
            prompt = tokenizer.decode(prompt_tokens)
            
            prompts.append(prompt)
            references.append(text)
        
        # Generate continuations
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_kwargs
            )
        
        # Decode generated texts
        generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]
        
        # Collect references and hypotheses for evaluation
        all_refs.extend(references)
        all_hyps.extend(generated_texts)
    
    # Compute metrics
    metrics = compute_metrics(all_refs, all_hyps)
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "evaluation.log"))
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load generation config
    generation_kwargs = {}
    if args.config:
        config = load_config(args.config)
        generation_kwargs = config.get("generation", {})
    
    # Generate samples from prompts
    if args.prompts_file:
        prompts = load_prompts(args.prompts_file)
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = None
    
    generated_samples = generate_sample_texts(
        model,
        tokenizer,
        device,
        prompts=prompts,
        num_samples=args.num_samples,
        **generation_kwargs
    )
    
    # Save generated samples
    samples_output_path = os.path.join(args.output_dir, "generated_samples.txt")
    with open(samples_output_path, "w") as f:
        for i, sample in enumerate(generated_samples):
            f.write(f"=== Sample {i+1} ===\n")
            f.write(sample)
            f.write("\n\n")
    
    logger.info(f"Saved {len(generated_samples)} generated samples to {samples_output_path}")
    
    # Evaluate on test dataset if provided
    if args.test_dataset:
        test_dataset = load_test_dataset(args.test_dataset)
        
        metrics = evaluate_model_on_dataset(
            model,
            tokenizer,
            test_dataset,
            device,
            batch_size=args.batch_size,
            **generation_kwargs
        )
        
        # Save metrics
        metrics_output_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")
        
        logger.info(f"Saved metrics to {metrics_output_path}")


if __name__ == "__main__":
    main() 