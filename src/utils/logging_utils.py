#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities for the AI Tale model training repository.
"""

import logging
import os
import sys
from datetime import datetime


def setup_logging(
    log_file=None, 
    log_level=logging.INFO, 
    format_string="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file. If None, logs to console only.
        log_level: Logging level (default: INFO)
        format_string: Format string for log messages
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that works with tqdm progress bars.
    Prevents logging messages from breaking tqdm progress bars.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            from tqdm.auto import tqdm
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def log_experiment_info(config, log_dir=None):
    """
    Log experiment information for easy reference.
    
    Args:
        config: Configuration dictionary
        log_dir: Directory to save experiment info file
    """
    logger = get_logger("experiment_info")
    
    # Log basic experiment info
    logger.info(f"Experiment: {config.get('experiment_name', 'unnamed_experiment')}")
    logger.info(f"Model: {config.get('base_model', 'unknown')}")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log training parameters
    training_config = config.get("training", {})
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {training_config.get('num_train_epochs', 'N/A')}")
    logger.info(f"  Batch size: {training_config.get('per_device_train_batch_size', 'N/A')}")
    logger.info(f"  Learning rate: {training_config.get('learning_rate', 'N/A')}")
    
    # Write experiment info to file if log_dir is specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        info_file = os.path.join(log_dir, "experiment_info.txt")
        
        with open(info_file, "w") as f:
            f.write(f"Experiment: {config.get('experiment_name', 'unnamed_experiment')}\n")
            f.write(f"Model: {config.get('base_model', 'unknown')}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Training parameters:\n")
            for k, v in training_config.items():
                f.write(f"  {k}: {v}\n") 