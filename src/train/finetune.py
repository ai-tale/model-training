#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for story generation models.
"""

import argparse
import logging
import os
import sys
import yaml
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_metrics
from src.evaluate.evaluate_model import generate_sample_texts

# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model for story generation")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model identifier (overrides config if provided)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory (overrides config if provided)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, fewer epochs)"
    )
    args = parser.parse_args()
    return args


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_datasets(config):
    """Load and prepare datasets for training and evaluation."""
    dataset_config = config["dataset"]
    
    # Load datasets
    data_files = {
        "train": dataset_config["train_file"],
        "validation": dataset_config["validation_file"]
    }
    
    extension = dataset_config["train_file"].split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=None,
    )
    
    # Sample for debug mode
    if args.debug:
        max_train_samples = min(1000, len(raw_datasets["train"]))
        max_eval_samples = min(100, len(raw_datasets["validation"]))
        raw_datasets["train"] = raw_datasets["train"].select(range(max_train_samples))
        raw_datasets["validation"] = raw_datasets["validation"].select(range(max_eval_samples))
        logger.info(f"Debug mode: Using {max_train_samples} training samples and {max_eval_samples} validation samples")
    
    # Get the column containing the text
    column_names = raw_datasets["train"].column_names
    text_column_name = dataset_config["text_column"] if dataset_config["text_column"] in column_names else column_names[0]
    
    return raw_datasets, text_column_name


def tokenize_datasets(raw_datasets, tokenizer, config, text_column_name):
    """Tokenize datasets for model training."""
    max_length = config["max_length"]
    
    # Preprocessing function for tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            padding="max_length" if dataset_config["line_by_line"] else False,
            truncation=True,
            max_length=max_length,
        )
    
    # Apply tokenization
    dataset_config = config["dataset"]
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=dataset_config["preprocessing_num_workers"],
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=not dataset_config["overwrite_cache"],
        desc="Tokenizing datasets",
    )
    
    return tokenized_datasets


def train(args, config):
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(config["seed"])
    
    # Override config with command line arguments if provided
    if args.model:
        config["base_model"] = args.model
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Save the effective config
    with open(os.path.join(config["output_dir"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Load tokenizer
    tokenizer_name = config["tokenizer_name"] if config["tokenizer_name"] else config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=config["use_fast_tokenizer"],
    )
    
    # Load datasets
    raw_datasets, text_column_name = prepare_datasets(config)
    
    # Tokenize datasets
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, config, text_column_name)
    
    # Load model
    logger.info(f"Loading model {config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16 if config["training"]["fp16"] else None,
    )
    
    # Resize token embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Prepare training arguments
    train_config = config["training"]
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    
    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_config["per_device_train_batch_size"],
        collate_fn=default_data_collator,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=train_config["per_device_eval_batch_size"],
        collate_fn=default_data_collator,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(train_config["adam_beta1"], train_config["adam_beta2"]),
        eps=train_config["adam_epsilon"],
        weight_decay=train_config["weight_decay"],
    )
    
    # Learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader) // train_config["gradient_accumulation_steps"]
    max_train_steps = train_config["num_train_epochs"] * num_update_steps_per_epoch
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * train_config["warmup_ratio"]),
        num_training_steps=max_train_steps,
    )
    
    # Setup wandb if enabled
    if config["logging"]["use_wandb"]:
        import wandb
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            name=config["experiment_name"],
            config=config,
        )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Enable mixed precision training if requested
    scaler = None
    if train_config["fp16"]:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    
    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {train_config['num_train_epochs']}")
    logger.info(f"  Batch size per device = {train_config['per_device_train_batch_size']}")
    logger.info(f"  Gradient Accumulation steps = {train_config['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    tr_loss = 0.0
    best_eval_loss = float("inf")
    
    progress_bar = tqdm(range(max_train_steps))
    
    for epoch in range(train_config["num_train_epochs"]):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with optional mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                outputs = model(**batch)
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / train_config["gradient_accumulation_steps"]
            
            # Backward pass with optional mixed precision
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (step + 1) % train_config["gradient_accumulation_steps"] == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["max_grad_norm"])
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
            
            # Track loss
            tr_loss += loss.item()
            epoch_loss += loss.item()
            
            # Logging
            if global_step % train_config["logging_steps"] == 0:
                avg_loss = tr_loss / global_step
                logger.info(f"Step {global_step}: train_loss = {avg_loss}")
                if config["logging"]["use_wandb"]:
                    wandb.log({"train_loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]})
            
            # Evaluation
            if global_step % train_config["eval_steps"] == 0:
                eval_results = evaluate(model, eval_dataloader, device, config)
                logger.info(f"Step {global_step}: eval_loss = {eval_results['eval_loss']}")
                
                if config["logging"]["use_wandb"]:
                    wandb.log(eval_results)
                
                # Save best model
                if eval_results["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_results["eval_loss"]
                    logger.info(f"New best eval loss: {best_eval_loss}")
                    model.save_pretrained(os.path.join(config["output_dir"], "best_model"))
                    tokenizer.save_pretrained(os.path.join(config["output_dir"], "best_model"))
            
            # Save checkpoint
            if global_step % train_config["save_steps"] == 0:
                output_dir = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                
                # Generate sample texts and log them
                if config["logging"]["use_wandb"]:
                    samples = generate_sample_texts(
                        model, 
                        tokenizer, 
                        device,
                        num_samples=3,
                        **config["generation"]
                    )
                    wandb.log({"generated_samples": [wandb.Html(s.replace("\n", "<br>")) for s in samples]})
            
            if global_step >= max_train_steps:
                break
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch}: avg_loss = {avg_epoch_loss}")
    
    # Save final model
    logger.info("Training complete!")
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    
    # Final evaluation
    final_eval_results = evaluate(model, eval_dataloader, device, config)
    logger.info(f"Final evaluation: {final_eval_results}")
    
    # Generate and log some final samples
    if config["logging"]["use_wandb"]:
        samples = generate_sample_texts(
            model, 
            tokenizer, 
            device,
            num_samples=5,
            **config["generation"]
        )
        wandb.log({"final_samples": [wandb.Html(s.replace("\n", "<br>")) for s in samples]})
    
    return final_eval_results


def evaluate(model, eval_dataloader, device, config):
    """Evaluate the model on the evaluation dataset."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss.item())
    
    eval_loss = np.mean(losses)
    eval_perplexity = np.exp(eval_loss)
    
    return {
        "eval_loss": eval_loss,
        "eval_perplexity": eval_perplexity,
    }


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(f"finetune_{timestamp}.log")
    
    # Load configuration
    config = load_config(args.config)
    
    # Start training
    train(args, config) 