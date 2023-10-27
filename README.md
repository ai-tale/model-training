# AI Tale Model Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)](https://pytorch.org/)

This repository contains training pipelines and fine-tuning scripts for AI Tale's specialized story generation models.

## 🔍 Overview

[AI Tale](https://aitale.tech/) is a platform that uses large language models to generate fairy tales and illustrated online storybooks. This repository houses the model training infrastructure that powers our story generation capabilities.

## 🌟 Features

- Fine-tuning pipelines for large language models specialized in storytelling
- Data processing utilities for narrative datasets
- Evaluation metrics for story coherence, creativity, and engagement
- Hyperparameter optimization tools
- Integration with our production inference API

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ai-tale/model-training.git
cd model-training

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📚 Usage

### Data Preparation

```bash
python src/data/prepare_dataset.py --config configs/data_config.yaml
```

### Model Fine-tuning

```bash
python src/train/finetune.py --model gpt2-medium --config configs/finetune_config.yaml
```

### Evaluation

```bash
python src/evaluate/evaluate_model.py --model-path models/storyteller-v1 --test-dataset data/test_stories.json
```

## 📁 Repository Structure

```
model-training/
├── configs/            # Configuration files for training, data processing
├── data/               # Dataset storage (gitignored, populated during setup)
├── docs/               # Documentation
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── scripts/            # Utility scripts for automation
├── src/                # Source code
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architecture definitions
│   ├── train/          # Training scripts and routines
│   ├── utils/          # Utility functions
│   └── evaluate/       # Evaluation metrics and benchmarks
└── tests/              # Unit and integration tests
```

## 🔬 Research

Our models and approaches are based on state-of-the-art research in narrative generation and creative text production. We're continuously refining our approaches based on the latest advancements in the field.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

- Alexander Monash (Founder & CEO): [GitHub](https://github.com/morfun95)
- AI Tale Team: [Website](https://aitale.tech)

## 🙏 Acknowledgements

- The Hugging Face team for their transformers library
- The PyTorch community
- All contributors and storytellers who have helped shape this project
