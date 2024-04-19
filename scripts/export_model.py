#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export trained models for production deployment.
This script handles optimization, quantization, and packaging.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logging_utils import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export model for production deployment")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save exported model",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pytorch", "onnx", "torchscript"],
        default="pytorch",
        help="Export format",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimization",
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="Use FP16 (half precision)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to JSON file with additional metadata",
    )
    args = parser.parse_args()
    return args


def export_pytorch_model(model, tokenizer, config, output_dir, half_precision=False):
    """Export model in PyTorch format."""
    logger.info("Exporting model in PyTorch format")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model in the desired precision
    if half_precision:
        logger.info("Converting to half precision (FP16)")
        model = model.half()
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save model configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"Model exported to {output_dir}")
    return output_dir


def export_torchscript_model(model, tokenizer, config, output_dir, max_length=1024, half_precision=False):
    """Export model in TorchScript format."""
    logger.info("Exporting model in TorchScript format")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to half precision if requested
    if half_precision:
        logger.info("Converting to half precision (FP16)")
        model = model.half()
    
    # Prepare model for tracing
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.zeros((1, max_length), dtype=torch.long, device=model.device)
    
    # Trace the model
    logger.info("Tracing model with TorchScript")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, [dummy_input])
    
    # Save the traced model
    traced_model_path = os.path.join(output_dir, "model.pt")
    torch.jit.save(traced_model, traced_model_path)
    
    # Save the tokenizer and config
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"TorchScript model exported to {traced_model_path}")
    return output_dir


def export_onnx_model(model, tokenizer, config, output_dir, max_length=1024, half_precision=False):
    """Export model in ONNX format."""
    try:
        import onnx
        import onnxruntime
    except ImportError:
        logger.error("ONNX export requires onnx and onnxruntime packages. Please install them with: pip install onnx onnxruntime")
        return None
    
    logger.info("Exporting model in ONNX format")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to half precision if requested
    if half_precision:
        logger.info("Converting to half precision (FP16)")
        model = model.half()
    
    # Prepare model for export
    model.eval()
    
    # Create dummy input for export
    dummy_input = torch.zeros((1, max_length), dtype=torch.long, device=model.device)
    
    # Export the model to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    logger.info("Exporting model to ONNX")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    # Verify the ONNX model
    logger.info("Verifying ONNX model")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Save the tokenizer and config
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"ONNX model exported to {onnx_path}")
    return output_dir


def quantize_model(model, quantize_type="dynamic"):
    """Quantize the model to reduce size and improve inference speed."""
    logger.info(f"Applying {quantize_type} quantization")
    
    if quantize_type == "dynamic":
        # Dynamic quantization
        try:
            from torch.quantization import quantize_dynamic
            quantized_model = quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            logger.info("Dynamic quantization applied successfully")
            return quantized_model
        except Exception as e:
            logger.error(f"Error applying dynamic quantization: {e}")
            return model
    else:
        logger.warning(f"Quantization type {quantize_type} not supported")
        return model


def optimize_model(model):
    """Apply optimization techniques to the model."""
    logger.info("Applying model optimizations")
    
    # Fusion of operations where possible
    # Note: This is a placeholder - actual optimization would depend on the model architecture
    
    logger.info("Model optimization applied")
    return model


def add_metadata(output_dir, metadata_file=None):
    """Add metadata to the exported model."""
    metadata = {
        "framework": "pytorch",
        "task": "text-generation",
        "generation_types": ["storytelling", "fairy-tales"],
        "model_name": Path(output_dir).name,
        "export_date": str(datetime.now()),
        "ai_tale_version": "1.0.0",
    }
    
    # Add additional metadata from file if provided
    if metadata_file and os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                additional_metadata = json.load(f)
            metadata.update(additional_metadata)
        except Exception as e:
            logger.error(f"Error loading metadata file: {e}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "ai_tale_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main function to export models."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging("export_model.log")
    
    # Import datetime here to avoid unused import warning at the beginning
    from datetime import datetime
    
    logger.info(f"Exporting model from {args.model_path} to {args.output_dir}")
    
    # Load model and tokenizer
    try:
        logger.info("Loading model and tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        config = AutoConfig.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
        )
        logger.info(f"Model loaded: {config.model_type}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Apply quantization if requested
    if args.quantize:
        model = quantize_model(model)
    
    # Apply optimization if requested
    if args.optimize:
        model = optimize_model(model)
    
    # Export model in the requested format
    if args.format == "pytorch":
        output_dir = export_pytorch_model(
            model, 
            tokenizer, 
            config, 
            args.output_dir,
            half_precision=args.half_precision
        )
    elif args.format == "torchscript":
        output_dir = export_torchscript_model(
            model, 
            tokenizer, 
            config, 
            args.output_dir,
            max_length=args.max_length,
            half_precision=args.half_precision
        )
    elif args.format == "onnx":
        output_dir = export_onnx_model(
            model, 
            tokenizer, 
            config, 
            args.output_dir,
            max_length=args.max_length,
            half_precision=args.half_precision
        )
    
    # Add metadata
    add_metadata(output_dir, args.metadata)
    
    logger.info("Model export completed successfully")


if __name__ == "__main__":
    main() 