**Contextual Real-time Intelligent Typo Identifier & Corrector (CRITIC)**

## Description
A context-aware spell-check ML System that provides appropriate feedback to the user in real-time after they make an error.
---

## Installation Instructions

### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.11 or newer

### Installation
Follow these steps to install and set up the project:

1. Clone the repository
2. Use of the precomputed data requires no additional installations, but if complete replication is desired the data should be downloaded from https://userinterfaces.aalto.fi/136Mkeystrokes/ and unpacked to the `data/` folder: the `scripts/` folder then has the processing. This takes a long time.
3. Install llama-cpp-python with these instructions: https://llama-cpp-python.readthedocs.io/en/latest/#installation-configuration
4. Open a terminal in where you cloned the repository to
5. Install JAX - ```pip install jax```
6. Install the following - ```pip install scipy tensorflow tensorflow-probability keras pandas```
7. To install the CRITIC module - ```pip install --editable .```
8. Download the Llama 3.2 model to the ```models/``` directory - https://huggingface.co/jxtngx/Meta-Llama-3.2-1B-Q4_K_M-GGUF/resolve/main/llama-3.2-1b-q4_k_m.gguf
9. To test the installation, a sample correction can be run using ```python src/critic/llama.py```
10. Evaluation of the model is done in `notebooks/evaluation.ipynb`
11. Training of the keyboard model and other processing is done in `notebooks/`
