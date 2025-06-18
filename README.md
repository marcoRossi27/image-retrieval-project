# image-retrieval-project
🚀 Image Retrieval Project with CLIP and Deep Metric Learning
This repository contains a complete implementation of an image retrieval system based on a CLIP (ViT-L/14) model, fine-tuned using Deep Metric Learning techniques. The project is designed for flexibility, allowing the use of either a default dataset (Food101) or a custom, user-provided dataset.

The multi-task training strategy combines several loss functions to learn highly discriminative image embeddings:

ProxyAnchorLoss: A robust proxy-based loss for metric learning.
TripletMarginLoss with Hard Negative Mining: To focus training on the most informative examples.
CrossEntropyLoss: Used as an auxiliary loss to regularize and stabilize the training process.
&lt;br>

📂 Project Structure
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
&lt;br>

⚙️ Setup and Installation
Follow these steps to get the project running on your local machine.

[!IMPORTANT]
Prerequisites:

Python 3.8+
Git
Procedure:

Clone the repository:

Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
 Navigate into the project directory:

Bash
cd YOUR_REPOSITORY_NAME
  ```

 (Recommended) Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
 Install the required dependencies:

Bash
pip install -r requirements.txt
 &lt;br>

▶️ Usage
The script can be run in two distinct modes.

Mode 1: Default Dataset (Food101)

This is the simplest way to run the project.

[!NOTE]
On the first run, the script will automatically download the Food101 dataset and organize it into the required folder structure. This might take some time.

To start the training with the default setup, simply run:

Bash
python main.py
Mode 2: Custom Dataset

You can use your own dataset, as long as it follows the required folder structure.

[!TIP]
Required Folder Structure:

your_dataset/
├── training/
│   ├── class1/
│   └── class2/
└── test/
    ├── query/
    └── gallery/
 To use your custom dataset, run the script with the --data_dir argument:

Bash
python main.py --data_dir /path/to/your/dataset
&lt;br>

🔧 Configuration
All major hyperparameters are centralized in the config.py file for easy experimentation.

[!NOTE]
Modifying config.py is the recommended way to test different settings without altering the core logic. You can easily tweak parameters such as:

LEARNING_RATE, BATCH_SIZE, EPOCHS
EMBED_DIM, UNFREEZE_LAYERS
PROXY_MARGIN, TRIPLET_MARGIN
