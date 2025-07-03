# image-retrieval-project
#  Image Retrieval with CLIP and Deep Metric Learning

## Abstract  
This repository presents a modular, end-to-end implementation of an image retrieval system based on a pre-trained CLIP ViT-L/14 backbone, fine-tuned via Deep Metric Learning (DML). Our multi-task training strategy combines ProxyAnchorLoss, TripletMarginLoss with hard-negative mining, and CrossEntropyLoss to learn highly discriminative embeddings. The code supports both the default Food101 dataset and any custom, user-supplied dataset in an ImageFolder format.

---

## 📂 Project Structure  
```text
.
├── main.py             # Entrypoint: training & evaluation
├── config.py   # Central hyperparameters & paths
├── requirements.txt    # Python dependencies
└── src/
    ├── data/           # Data preparation
    │   └── datasets.py
    ├── transforms/     # Data augmentations
    │   └── transforms.py
    ├── samplers/       # Metric-learning samplers
    │   └── samplers.py
    ├── loader/         # Data loaders
    │   └── loader.py
    ├── models/         # Model definitions
    │   └── model.py
    ├── training/       # Training loop & utilities
    │   └── trainer.py
    └── inference/      # Embedding extraction & evaluation
        └── evaluate.py
```
⚙️ Installation
1. Prerequisites
  - Python 3.8 or higher
  - Git
2. Clone & Virtual Environment
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```
3. Dependencies
```bash
pip install -r requirements.txt
```
▶️ Usage
1. Default Dataset (Food101)
On first execution, the script will download and preprocess Food101 automatically.
```bash
python main.py --mode train
python main.py --mode eval
```
2. Custom Dataset
Your dataset must follow an ImageFolder-style hierarchy:
```bash
your_dataset/
├── training/
│   ├── class1/
│   └── class2/
└── test/
    ├── query/
    └── gallery/
```
Invoke with:
```bash
python main.py --mode train --data_dir /path/to/your_dataset
python main.py --mode eval  --data_dir /path/to/your_dataset
```
🔧 Configuration
```text
## 🔧 Configuration

All hyperparameters and paths live in `config.py`. You can tweak these to suit your dataset or hardware.

| Parameter             | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `DEVICE`              | Compute device, either `"cuda"` (GPU) or `"cpu"`.               |
| `NUM_CLASSES`         | Number of classes to sample (default: `100`).                   |
| `MAX_PER_CLASS`       | Maximum images per class (default: `250`).                      |
| `BATCH_SIZE`          | Mini-batch size (default: `32`).                                |
| `LR_BASE` / `LR_BACKBONE` | Learning rates for the MLP head and the CLIP backbone.    |
| `EPOCHS` / `WARMUP_EPOCHS` | Total training epochs and warm-up epochs.              |
| `EMBED_DIM`           | Dimensionality of the embedding vector (default: `2048`).       |
| `MARGIN` / `ALPHA`    | Margin (for triplet loss) and alpha (for ProxyAnchorLoss).      |
| `CE_WEIGHT`           | Weight of the `CrossEntropyLoss` term in the total loss.        |
| `dataset_name`        | `"Food101"` (default) or `"Custom"` (for your own ImageFolder). |

> **Tip:** Experiment with these settings for different domains and compute budgets!  
```
