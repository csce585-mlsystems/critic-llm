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

1. Clone the Repository
2. Install llama-cpp-python with these instructions: https://llama-cpp-python.readthedocs.io/en/latest/#installation-configuration
3. Open a terminal in where you cloned the repository to
4. Install JAX - ```pip install jax```
5. Install the following - ```pip install scipy tensorflow tensorflow-probability keras pandas```
6. To install the CRITIC module - ```pip install --editable .```
7. Download the Llama 3.2 model to the ```models/``` directory - https://huggingface.co/jxtngx/Meta-Llama-3.2-1B-Q4_K_M-GGUF/resolve/main/llama-3.2-1b-q4_k_m.gguf
8. CRITIC can then be ran with ```python src/critic/llama.py```

