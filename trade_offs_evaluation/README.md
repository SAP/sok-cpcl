# Privacy-Accuracy-Performance Trade-offs Evaluation

This folder contains the set of experiments to evaluate accuracy and performance trade-offs between centralized learning, federated learning (FL), and outsourced learning (OL) with and without differential privacy (DP).

## Requirements and Setup

### 1. Install Base Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Modified CrypTen Library

The `Crypten/` folder contains a custom CrypTen library with integrated DP optimizers and noise sampling:

```bash
cd Crypten
python setup.py install
cd ..
```

## Folder Structure

- **`Crypten/`** — Modified CrypTen library with DP extensions

- **`dataset/`** — Data loading and preprocessing
  - `mnist/` — MNIST dataset utilities
  - `fashion_mnist/` — Fashion-MNIST dataset utilities

- **`utils/`** — Shared utilities for all experiments
  - `models.py` — PyTorch and CrypTen model definitions (`ThreeLayerNN`, `CryptenThreeLayerNN`)
  - `mpc_dpsgd_trainer.py` — Main trainer class for encrypted DP-SGD training
  - `eval_utils.py` — Evaluation utilities, timing tracking, configuration
  - `metrics.py` — Classification metrics (accuracy, cross-entropy loss)

- **`performance_eval/`** — Performance benchmarking scripts (see [performance_eval/README.md](performance_eval/README.md) for details)
  - `mpc_noise.py` / `mpc_noise_launcher.py` — Noise sampling overhead evaluation
  - `ol_evaluation.py` / `ol_launcher.py` — Oblivious Learning performance
  - `fl_evaluation.py` — Federated Learning performance

- **`*.ipynb`** — Accuracy evaluation notebooks

## Accuracy Evaluation Notebooks

These notebooks train models under different privacy and learning settings:

### Non-DP Baseline
- **`clear_non_dp_training.ipynb`** — Centralized, non-private training on plaintext data

### DP Baseline (Centralized Training)
- **`clear_dp_training.ipynb`** — Centralized DP-SGD training

### Federated Learning
- **`fl_non_dp_training.ipynb`** — Federated Learning without DP
- **`fl_dp_training.ipynb`** — Federated Learning with DP Federated Averaging
- **`local_dp_fl_training.ipynb`** — Local DP applied per-client in federated setting

It is possible to evaluate the effect of collusion with partial noise aggregation by setting the `collusion_factor` which is the square root of the number of colluding clients.

### Outsourced Learning
- **`ol_non_dp_training.ipynb`** — Outsourced Learning without DP
- **`ol_dp_training.ipynb`** — Outsourced Learning with DP-SGD


## DP training with CrypTen
To implement DP training with CrypTen we modified the SGD optimizer and provided sampling utilities for DP noise.

### DP Optimizers
- **DP-SGD** (`crypten/optim/dpsgd.py`) — Differentially private SGD for encrypted training
- **DP-AdamW** (`crypten/optim/dpadamw.py`) — Differentially private Adam optimizer variant

### DP Utilities
- **Gaussian Noise Sampling** (`crypten/dp/gaussian_noise.py`) — Local and global Gaussian DP noise
- **Skellam Noise Sampling** (`crypten/dp/skellam_noise.py`) — Skellam distribution (note: global sampling incomplete for MPC)
- **Privacy Accounting** (`crypten/dp/rdp_accountant.py`) — Renyi differential privacy tracking
- **DP Utils** (`crypten/dp/dp_utils.py`) — Gradient clipping, noise sampling, privacy accounting helpers


## Important Notes
- **CrypTen Installation**: Must be installed from source in `Crypten/` folder before running notebooks
- **Performance Evaluation**: Performance benchmarks should run on AWS instances with sufficient GPU/CPU resources (see `performance_eval/README.md`)
