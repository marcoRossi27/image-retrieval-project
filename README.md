# image-retrieval-project
Image Retrieval Project with CLIP and Deep Metric Learning
This repository contains an implementation of an image retrieval system based on a CLIP (ViT-L/14) model, fine-tuned using Deep Metric Learning techniques. The project is designed to be flexible, allowing the use of either a default dataset (Food101) or a custom user-provided dataset.

The multi-task training strategy combines several loss functions to learn highly discriminative image embeddings:

ProxyAnchorLoss: A robust proxy-based loss for metric learning.
TripletMarginLoss with Hard Negative Mining: To focus training on the most informative and difficult examples.
CrossEntropyLoss: Used as an auxiliary loss to regularize and stabilize the training process.
Project Structure
The code is organized into a modular structure to ensure clarity and maintainability:

.
├── main.py             # Main script to launch training and evaluation
├── config.py           # Central configuration file for all hyperparameters
├── requirements.txt    # Python dependencies for the project
└── src/
    ├── data/           # Modules for data loading and preparation
    │   └── loader.py
    ├── training/       # Modules for model definition and training
    │   ├── model.py
    │   └── trainer.py
    └── inference/      # Modules for evaluation and inference
        └── evaluate.py
Setup and Installation
To run this project locally, follow these steps.

Prerequisites:

Python 3.8+
Git
Procedure:

Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
 Navigate into the project directory:

Bash
cd YOUR_REPOSITORY_NAME
 (Recommended) Create a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
 Install the dependencies:

Bash
pip install -r requirements.txt
Usage
The script can be run in two modes.

Mode 1: Running with the Default Dataset (Food101)

This is the default mode. On the first run, the script will automatically download the Food101 dataset and organize it into the required structure.

Bash
python main.py
Mode 2: Running with a Custom Dataset

You can use your own dataset, provided it follows the structure below:

your_dataset/
├── training/
│   ├── class1/
│   │   └── image1.jpg
│   └── class2/
│       └── image2.jpg
└── test/
    ├── query/
    │   └── query_image1.jpg
    └── gallery/
        └── gallery_image1.jpg
To use your custom dataset, run the script and specify the path with the --data_dir argument:

Bash
python main.py --data_dir /path/to/your/dataset
Configuration
All major hyperparameters for the project can be modified in the config.py file. This includes:

LEARNING_RATE, BATCH_SIZE, EPOCHS
Model parameters like EMBED_DIM and UNFREEZE_LAYERS
Loss function parameters (PROXY_MARGIN, TRIPLET_MARGIN, etc.)
This design allows for easy experimentation with different configurations without altering the source code.
