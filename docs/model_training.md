# AI Tale Model Training Documentation

This document provides detailed information about training the story generation models used in the AI Tale platform.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Best Practices](#best-practices)

## Overview

The AI Tale platform generates fairy tales and illustrated online storybooks using specialized language models. These models are fine-tuned from pre-trained language models (like GPT-2, T5, etc.) on custom datasets of fairy tales and children's stories.

The training pipeline consists of several key steps:
1. Dataset preparation and preprocessing
2. Model fine-tuning with appropriate hyperparameters
3. Evaluation using both automated metrics and human evaluation
4. Model optimization for production deployment

## Model Architecture

Our story generation models are based on transformer architectures, primarily using autoregressive (GPT-style) models for generation.

### Base Models

We typically start with the following pre-trained models:
- GPT-2 (124M, 355M, 774M parameters)
- GPT-Neo (1.3B parameters)
- T5 (for specific conditional generation tasks)

### Customizations

We apply several modifications to the base architecture:
- Additional special tokens for story structure (beginning, ending, etc.)
- Specialized tokenization patterns for narrative elements
- Task-specific heads for emotion/theme classification when needed

## Dataset Preparation

### Data Sources

Our training datasets are compiled from various sources:
- Public domain fairy tales (Brothers Grimm, Hans Christian Andersen, etc.)
- Children's stories with appropriate licenses
- Custom-written stories specifically for AI Tale
- Aesop's fables and other classic short stories

### Preprocessing

The `src/data/prepare_dataset.py` script handles dataset preparation with the following steps:
1. Loading raw data from various sources
2. Cleaning and normalizing text (removing headers, footers, etc.)
3. Filtering inappropriate content
4. Adding special tokens for story structure
5. Augmenting the dataset through techniques like:
   - Paraphrasing
   - Style transfer
   - Adding varied story beginnings/endings
6. Splitting into train/validation/test sets

### Sample Command

```bash
python src/data/prepare_dataset.py \
    --config configs/data_config.yaml \
    --output-dir data/processed
```

## Training Process

### Fine-tuning Approach

We use a multi-stage fine-tuning approach:
1. Initial broad fine-tuning on the entire dataset
2. Specialized fine-tuning for specific age groups or themes
3. Targeted fine-tuning for specific narrative styles

### Hyperparameters

Key hyperparameters for our models include:
- Learning rate: 5e-5 (with linear decay)
- Batch size: 4-8 per device (depending on model size)
- Training epochs: 3-5 (with early stopping)
- Maximum sequence length: 1024 tokens
- Gradient accumulation steps: 8 (for effective larger batch sizes)

These parameters can be configured in the `configs/finetune_config.yaml` file.

### Sample Command

```bash
python src/train/finetune.py \
    --config configs/finetune_config.yaml \
    --model gpt2-medium
```

### Tracking Experiments

We use Weights & Biases for experiment tracking, which records:
- Training and validation loss
- Evaluation metrics
- Sample generated stories
- Resource utilization

## Evaluation

### Automated Metrics

We evaluate our models using both standard NLP metrics and custom narrative-specific metrics:

**Standard Metrics:**
- Perplexity
- BLEU score
- ROUGE score

**Custom Metrics:**
- Narrative coherence
- Emotional arc consistency
- Character consistency
- Age-appropriateness
- Creative language usage

### Human Evaluation

In addition to automated metrics, we conduct human evaluations focusing on:
- Story engagement
- Plot coherence
- Character development
- Age-appropriate content
- Educational value
- Entertainment value

### Sample Command

```bash
python src/evaluate/evaluate_model.py \
    --model-path models/storyteller-v1 \
    --test-dataset data/processed/fairy_tales_test.json \
    --output-dir evaluation_results
```

## Deployment

### Model Optimization

Before deployment, we optimize our models through:
1. Quantization (INT8 or mixed precision)
2. Distillation (where appropriate)
3. ONNX conversion for certain deployment targets
4. TorchScript compilation

### Integration with Production API

Our models are deployed via:
1. Model packaging with version control
2. Docker containerization
3. API endpoint creation with batching support
4. Caching for common requests

## Best Practices

### Training Tips

- Start with a smaller model for initial experiments
- Use gradient accumulation for effectively larger batch sizes
- Monitor validation loss to prevent overfitting
- Generate samples periodically during training to check quality
- Save checkpoints frequently (every 1000 steps)

### Story Generation Settings

For optimal story generation, we recommend:
- Temperature: 0.7-0.9
- Top-k: 50
- Top-p (nucleus sampling): 0.9
- No repeat ngram size: 3
- Min length: Based on desired story length (typically 200+ tokens)
- Max length: 1024 tokens

These parameters can be adjusted in the generation section of the configuration file.

## Conclusion

This training infrastructure provides a robust foundation for creating specialized story generation models. By following this documentation, you should be able to train, evaluate, and deploy your own storytelling models for the AI Tale platform. 