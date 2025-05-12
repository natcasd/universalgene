# Gene Expression Representation Project

This repository contains implementations of five architectures: `cdan`, `transformer_dann`, `dann`, and two contrastive models. Below are the steps to preprocess the gene expression data, train the models, and generate visualizations.

---

## Prerequisites

Ensure you have installed all required dependencies. You can do so by running:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Use `src/data_download.py` to download gene expression data to your local src/data folder. Dataset is ~6gb.

---

## Data Preprocessing

1. **Run `data_preprocess.ipynb`**  
   This notebook preprocesses the raw data and prepares it for training. Open the notebook and execute all cells.

2. **Run `dann_preprocess.ipynb`**  
   This notebook performs additional preprocessing specific to the `dann`-based architectures. Open the notebook and execute all cells.

---

## Training the Models

### 1. DANN, CDAN, Transformer DANN (TDAN) Architectures

To train these architectures, use the `dann_training.py` script. The architecture to train can be specified using the `--model` argument. For example:

- **CDAN**:

  ```bash
  python dann_training.py --model cdan
  ```

- **Transformer DANN**:

  ```bash
  python dann_training.py --model transformer_dann
  ```

- **DANN**:
  ```bash
  python dann_training.py --model dann
  ```

### 2. Contrastive Models

To train the contrastive models, use the `train_contrastive.py` script. For example:

```bash
python train_contrastive.py
```

---

## Generating Visualizations

### 1. DANN-Based Architectures

To generate visualizations for `cdan`, `transformer_dann`, or `dann`, use the `cdan_visualization.ipynb` notebook.

### 2. Contrastive Models

To generate visualizations for the contrastive models, use the `contrastive_visualization_FINAL.ipynb` notebook.

---

## Output

- Trained model checkpoints will be saved in the `checkpoints/` directory.
- Visualizations will be saved in the `figures/` directory.
