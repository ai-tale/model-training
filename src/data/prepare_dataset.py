#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preparation script for fairy tale datasets.
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logging_utils import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare datasets for training fairy tale generation models")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory (overrides config if provided)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process fewer examples)"
    )
    args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(source_config):
    """Load raw data from source files."""
    all_data = []
    
    # Process fairy tales
    for source in source_config.get("fairy_tales", []):
        try:
            logger.info(f"Loading data from {source['name']} at {source['path']}")
            
            with open(source["path"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Add source information
            for item in data:
                item["source"] = source["name"]
                item["weight"] = source.get("weight", 1.0)
            
            all_data.extend(data)
            logger.info(f"Loaded {len(data)} items from {source['name']}")
            
        except Exception as e:
            logger.error(f"Error loading data from {source['path']}: {e}")
    
    # Process custom stories
    for source in source_config.get("custom_stories", []):
        try:
            logger.info(f"Loading data from {source['name']} at {source['path']}")
            
            with open(source["path"], "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Add source information
            for item in data:
                item["source"] = source["name"]
                item["weight"] = source.get("weight", 1.0)
            
            all_data.extend(data)
            logger.info(f"Loaded {len(data)} items from {source['name']}")
            
        except Exception as e:
            logger.error(f"Error loading data from {source['path']}: {e}")
    
    logger.info(f"Loaded {len(all_data)} items in total")
    return all_data


def clean_and_filter_data(data, config):
    """Clean and filter the data based on configuration."""
    preprocessing_config = config["preprocessing"]
    
    min_length = preprocessing_config.get("min_story_length", 100)
    max_length = preprocessing_config.get("max_story_length", 10000)
    remove_headers = preprocessing_config.get("remove_headers", True)
    remove_footers = preprocessing_config.get("remove_footers", True)
    
    cleaned_data = []
    
    for item in tqdm(data, desc="Cleaning data"):
        try:
            # Get the story text
            text = item.get("text", "")
            
            # Clean up text
            if remove_headers:
                # Remove headers (patterns like "Chapter X", "Story Title", etc.)
                text = re.sub(r"^(Chapter|CHAPTER|Story|STORY).*?\n\n", "", text)
            
            if remove_footers:
                # Remove footers (patterns like "THE END", "End of story", etc.)
                text = re.sub(r"\n\n(THE END|The End|End of Story|End)$", "", text)
            
            # Remove extra whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()
            
            # Check length
            if len(text) < min_length:
                logger.debug(f"Skipping story: too short ({len(text)} chars)")
                continue
                
            if len(text) > max_length:
                logger.debug(f"Truncating story: too long ({len(text)} chars)")
                text = text[:max_length]
            
            # Update the item
            item["text"] = text
            
            # Apply content filtering if enabled
            if preprocessing_config.get("adult_content_filter", False):
                # Simple keyword-based filtering (would use more sophisticated methods in production)
                adult_keywords = ["explicit", "sex", "erotic", "xxx", "pornographic"]
                if any(keyword in text.lower() for keyword in adult_keywords):
                    logger.debug(f"Skipping story: contains adult content")
                    continue
            
            # Apply violence filtering if enabled
            violence_threshold = preprocessing_config.get("violence_threshold", 1.0)
            if violence_threshold < 1.0:
                # Simple keyword-based detection (would use more sophisticated methods in production)
                violence_keywords = ["killed", "murdered", "slaughtered", "gore", "blood", "violent"]
                violence_score = sum(1 for keyword in violence_keywords if keyword in text.lower()) / len(violence_keywords)
                
                if violence_score > violence_threshold:
                    logger.debug(f"Skipping story: violence score {violence_score} exceeds threshold {violence_threshold}")
                    continue
            
            # Format text with special tokens if configured
            if preprocessing_config.get("add_special_tokens", False):
                story_starter = preprocessing_config.get("story_starter_token", "<|story|>")
                story_end = preprocessing_config.get("story_end_token", "<|endofstory|>")
                text = f"{story_starter}\n{text}\n{story_end}"
                item["text"] = text
            
            cleaned_data.append(item)
            
        except Exception as e:
            logger.error(f"Error processing item: {e}")
    
    logger.info(f"Cleaned data: {len(cleaned_data)} items (from {len(data)} raw items)")
    return cleaned_data


def split_data(data, config):
    """Split data into train, validation, and test sets."""
    preprocessing_config = config["preprocessing"]
    
    train_ratio = preprocessing_config.get("train_ratio", 0.8)
    val_ratio = preprocessing_config.get("val_ratio", 0.1)
    test_ratio = preprocessing_config.get("test_ratio", 0.1)
    
    # Shuffle data
    np.random.seed(42)  # Fixed seed for reproducibility
    np.random.shuffle(data)
    
    # Calculate split indices
    n = len(data)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)
    
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    
    logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }


def augment_data(data, config):
    """Augment data based on configuration."""
    preprocessing_config = config["preprocessing"]
    
    if not preprocessing_config.get("augmentation_enabled", False):
        logger.info("Data augmentation disabled")
        return data
    
    logger.info("Augmenting data...")
    
    paraphrase_ratio = preprocessing_config.get("paraphrase_ratio", 0.0)
    style_transfer_ratio = preprocessing_config.get("style_transfer_ratio", 0.0)
    
    augmented_data = data.copy()
    
    # Simple paraphrasing (in a production system, this would use a real paraphrasing model)
    if paraphrase_ratio > 0:
        paraphrase_count = int(len(data) * paraphrase_ratio)
        logger.info(f"Paraphrasing {paraphrase_count} items")
        
        for i in range(paraphrase_count):
            if i >= len(data):
                break
                
            # Create a copy of the item
            item = data[i].copy()
            
            # Simple "paraphrasing" by replacing some common words
            # (this is just a placeholder - real systems would use ML models)
            replacements = {
                "said": "exclaimed",
                "walked": "strolled",
                "looked": "gazed",
                "ran": "dashed",
                "big": "large",
                "small": "tiny",
                "happy": "joyful",
                "sad": "sorrowful"
            }
            
            text = item["text"]
            for old, new in replacements.items():
                text = re.sub(r'\b' + old + r'\b', new, text)
            
            item["text"] = text
            item["augmented"] = "paraphrase"
            
            augmented_data.append(item)
    
    # Simple style transfer (in a production system, this would use a real style transfer model)
    if style_transfer_ratio > 0:
        style_count = int(len(data) * style_transfer_ratio)
        logger.info(f"Style transferring {style_count} items")
        
        for i in range(style_count):
            if i >= len(data):
                break
                
            # Create a copy of the item
            item = data[i].copy()
            
            # Simple "style transfer" by adding some fantasy elements
            # (this is just a placeholder - real systems would use ML models)
            fantasy_elements = [
                "The magical forest whispered secrets as ",
                "Under the enchanted moon, ",
                "As if by wizardry, ",
                "The ancient spell began to work as ",
                "With fairy dust sparkling in the air, "
            ]
            
            text = item["text"]
            
            # Insert a fantasy element at the beginning of a random paragraph
            paragraphs = text.split("\n\n")
            if len(paragraphs) > 1:
                idx = np.random.randint(1, len(paragraphs))
                fantasy_element = np.random.choice(fantasy_elements)
                
                # Make the first letter lowercase if needed
                first_sentence = paragraphs[idx].split(". ")[0]
                if first_sentence and first_sentence[0].isupper():
                    paragraphs[idx] = fantasy_element + paragraphs[idx][0].lower() + paragraphs[idx][1:]
                else:
                    paragraphs[idx] = fantasy_element + paragraphs[idx]
                    
                text = "\n\n".join(paragraphs)
            
            item["text"] = text
            item["augmented"] = "style_transfer"
            
            augmented_data.append(item)
    
    logger.info(f"Augmented data: {len(augmented_data)} items (from {len(data)} original items)")
    return augmented_data


def save_split_data(split_data, config, output_dir):
    """Save data splits to files."""
    output_config = config["output"]
    
    if output_dir:
        processed_dir = output_dir
    else:
        processed_dir = output_config["processed_dir"]
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save each split
    for split_name, data in split_data.items():
        # Determine output filename
        if split_name == "train":
            output_file = os.path.join(processed_dir, output_config["train_file"])
        elif split_name == "validation":
            output_file = os.path.join(processed_dir, output_config["validation_file"])
        elif split_name == "test":
            output_file = os.path.join(processed_dir, output_config["test_file"])
        else:
            output_file = os.path.join(processed_dir, f"{split_name}.json")
        
        logger.info(f"Saving {len(data)} {split_name} examples to {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Save metadata
    metadata = {
        "splits": {
            "train": len(split_data["train"]),
            "validation": len(split_data["validation"]),
            "test": len(split_data["test"]),
        },
        "total": sum(len(data) for data in split_data.values()),
        "config": config,
    }
    
    metadata_file = os.path.join(processed_dir, output_config["metadata_file"])
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved metadata to {metadata_file}")


def main():
    """Main function to prepare datasets."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging("prepare_dataset.log")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load raw data
    data = load_raw_data(config["source"])
    
    # Sample a smaller subset for debug mode
    if args.debug:
        sample_size = min(100, len(data))
        logger.info(f"Debug mode: Sampling {sample_size} examples")
        np.random.seed(42)
        data = np.random.choice(data, size=sample_size, replace=False).tolist()
    
    # Clean and filter data
    cleaned_data = clean_and_filter_data(data, config)
    
    # Augment data (for train set only)
    augmented_data = augmented_data = cleaned_data
    
    # Split data
    split_data_dict = split_data(augmented_data, config)
    
    # Augment only the training data
    train_data = split_data_dict["train"]
    augmented_train_data = augment_data(train_data, config)
    split_data_dict["train"] = augmented_train_data
    
    # Save processed data
    save_split_data(split_data_dict, config, args.output_dir)
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main() 