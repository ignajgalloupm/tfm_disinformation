# Verification of Disinformation through Information Retrieval and Natural Language Inference

This repository contains the source code and implementation details for the Master's Thesis project focused on creating an efficient and adaptable system for automated fact verification. The system is designed to verify natural language claims against a large knowledge base (Wikipedia) using a dual-objective training process involving contrastive learning and binary cross-entropy.

The project is architected in Python using the PyTorch framework and leverages a Qdrant vector database for high-performance evidence retrieval.

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up Qdrant Vector Database](#2-set-up-qdrant-vector-database)
- [Running the Project](#running-the-project)
  - [Main Entry Point: `test.ipynb`](#main-entry-point-testipynb)
  - [Key Configuration](#key-configuration)

## Project Overview

The core objective of this project is to develop and validate an efficient fact-checking pipeline that addresses three key properties: **efficiency**, **adaptability**, and **interpretability**. Unlike monolithic language models, this system leverages an external, updatable knowledge base and a custom-designed Natural Language Inference (NLI) module that operates directly on vector embeddings.

The system is trained and evaluated on the **FEVER (Fact Extraction and VERification)** dataset.

## System Architecture

The verification process is split into two main stages: Evidence Retrieval and Claim Verification. The training architecture employs a dual-loss system to optimise both stages simultaneously.

![Training Architecture Diagram](https://github.com/ignajgalloupm/tfm_disinformation/blob/main/diagrams/Train_2%20(1).png)

- **Encoder Model:** A pre-trained transformer model (e.g., `gte-large-en-v1.5`), implemented in `embgen.py`, encodes claims and documents into high-dimensional vectors.
- **Vector Database:** A **Qdrant** database, managed by `vector_database.py`, stores the embeddings of the Wikipedia corpus for fast evidence retrieval.
- **NLI Module:** A custom, lightweight feed-forward neural network that takes the claim and evidence embeddings as input and predicts the final verdict (SUPPORTED or NOT SUPPORTED).

## Setup and Installation

Follow these steps to set up the environment and run the project.

### Prerequisites
- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/products/docker-desktop/)
- Python 3.9+
- An NVIDIA GPU with at least 24 GB of VRAM is recommended for training.

### 1. Clone the Repository
```bash
git clone [https://github.com/ignajgalloupm/tfm_disinformation.git](https://github.com/ignajgalloupm/tfm_disinformation.git)
cd tfm_disinformation
```

### 2. Set Up Qdrant Vector Database
This project uses a Docker container to run the Qdrant vector database.

To pull the Qdrant image and start the container, run the following command. This command exposes the correct port (`6333`) for the API and creates a named volume (`qdrant_storage`) to persist your database on your local machine.

```bash
docker run -p 6333:6333 \
    -v qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```
The Qdrant dashboard will now be accessible at `http://localhost:6333/dashboard`. The Python scripts are configured to connect to the API at this address.

## Running the Project

### Main Entry Point: `test.ipynb`
The primary way to run the training and evaluation pipeline is through the `test.ipynb` Jupyter Notebook. This notebook handles:
- Loading and preprocessing the datasets (`fever_dataset.py`, `wiki_dataset.py`).
- Initializing the models (`embgen.py`, NLI model).
- Setting up the vector databases (`vector_database.py`).
- Defining the optimiser and learning rate scheduler.
- Executing the main training and validation loops from `train2.py` and `validation2.py`.

To run the project, open and execute the cells in `test.ipynb`.

### Key Configuration
Hyperparameters and model configurations are set directly within the notebook. Key variables include:
- `N_EPOCHS`: Number of training epochs.
- `BATCH_SIZE`: Batch size for claims processing.
- `super_batch`: Corresponds to gradient accumulation steps.
- Learning rates for the encoder and NLI module in the `torch.optim.AdamW` definition.

Model checkpoints will be saved to and loaded from a `models/` directory, and evaluation metrics will be stored in a `metrics/` directory.

