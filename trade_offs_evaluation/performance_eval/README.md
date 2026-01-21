# Performance Evaluation

Benchmarking scripts to evaluating computational and communication overhead of different learning paradigms and DP noise techniques.

## Overview

This folder contains scripts to measure:
- **Noise sampling overhead**: Cost of sampling DP noise locally vs. globally in MPC
- **OL (Outsourced Learning) overhead**: Server-side communication and computation costs
- **FL (Federated Learning) overhead**: Client-side and server-side communication and computation costs

## Requirements and Setup
These scripts are designed to run on remote AWS instances. Ensure the instance has:
- Modified CrypTen library installed (from parent `Crypten/` folder)
- GPU support (optional but recommended)
- Sufficient CPU cores for multi-party computation

## Scripts

### Noise Sampling Evaluation

**Files**: `mpc_noise.py`, `mpc_noise_launcher.py`, `aws_launcher_noise.sh`

Measures the performance overhead of DP noise sampling with different sampling techniques

```bash
./aws_launcher_noise.sh
```

Evaluates:
- Local Gaussian noise sampling (in plaintext), i.e., `--noise_type local` inside `./aws_launcher_noise.sh`
- Global Gaussian noise sampling (in encrypted MPC), i.e., `--noise_type global` inside `./aws_launcher_noise.sh` 

### Outsourced Learning (OT) Evaluation

**Files**: `ol_evaluation.py`, `ol_launcher.py`, `aws_launcher_ol.sh`

Measures performance overhead of outsourced learning (using secure multi-party computation).

```bash
./aws_launcher_ol.sh
```

Evaluates:
- Forward pass latency
- Backward pass latency
- Communication rounds and bytes transferred
- Comparison with plaintext baseline

### Federated Learning Evaluation

**Files**: `fl_evaluation.py`

Measures client-side and server-side overhead in federated learning with DP.

```bash
python fl_evaluation.py --config config.yaml
```

Evaluates:
- Client-side gradient clipping and noise addition cost
- Server-side aggregation cost
- Communication overhead (gradient size, number of rounds)

### Training Utilities

**File**: `training.py`

Common training loop implementations used by evaluation scripts:
- Single-party training loop (plaintext baseline)
- Multi-party training loop (MPC/OL)
- Simulated federated training loop

## Output

Results are saved as JSON files in the `data/` subdirectory

## AWS Deployment

To run on AWS:

1. Prepare an EC2 instance with CPU/GPU and sufficient storage
2. Clone this repository and install dependencies
3. Run the launcher script:
   ```bash
   chmod +x aws_launcher_noise.sh
   ./aws_launcher_noise.sh
   ```
4. Results are saved locally and can be transferred back via SCP