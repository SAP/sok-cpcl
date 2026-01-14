# MPC Noise Sampling Experiments

This folder contains the code and scripts to run experiments for noise sampling and basic protocols using the MP-SPDZ framework.

## Structure
- `src` contains python and MP-SPDZ source code for noise sampling and basic protocols.
    - `python/` contains python scripts to sample noise from discrete distributions and utility scripts to compute distribution parameters from privacy parameters.
    - `spdz/` contains MP-SPDZ source code for noise sampling and basic protocols.
- `data/` contains input data files for MP-SPDZ programs.
- `utils/` contains utility scripts for building MP-SPDZ programs, running experiments, and parsing results.
    - `build_and_setup/` contains scripts to build MP-SPDZ and set up the environment.
    - `run_experiments/` contains example bash scripts to run MP-SPDZ programs.
    - `parsers/` contains python scripts to parse MP-SPDZ logs and compute statistics.


## Requirements, Setup and Usage

### Python scripts
The `src/python/` folder contains python script to sample noise from discrete distributions, specifically:
- `skellam.py`: samples from the Skellam distribution using the difference of two Poisson samples.
- `discrete_gaussian.py`: samples from the discrete Gaussian distribution using the rejection sampling method from tensorflow-privacy.
- `dgauss_exact.py`: samples from the discrete Gaussian distribution using the exact [method](https://arxiv.org/pdf/2004.00010.pdf). 

 The `src/python/utils/` folder contains scripts to computed distribution parameters from given privacy parameters using existing methods from the literature.

#### 1. Requirements
**Requirements**
- Python 3.8+ with requirements in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

#### 2. Run scripts
To run the noise sampling scripts, navigate to the `src/python/` folder and execute the desired script. For example, to sample Laplace noise:
```bash
cd src/python/
python3 laplace_sampling.py --epsilon 1.0 --delta 1e-5
```



### MP-SPDZ scripts
**Clone MP-SPDZ and install dependencies**

#### 1. Clone MP-SPDZ and check dependencies

Depending on your platform, follow one of the methods below to get MP-SPDZ built and ready in  the `utils/build_and_setup` folder.
1. Linux 
```bash
chmod +x scripts/build_mp_spdz_linux.sh
./scripts/build_mp_spdz_linux.sh
```
2. macOS
```bash
chmod +x scripts/build_mp_spdz_macos.sh
./scripts/build_mp_spdz_macos.sh
```

#### 2. Build MP-SPDZ programs from source

To build the required MP-SPDZ programs from source, run the following script in the `utils/build_and_setup` folder:
```bash
chmod +x utils/scripts/build_spdz_sources.sh
./utils/scripts/build_spdz_sources.sh
```

This script copies the source files from `src/spdz/` to the MP-SPDZ `Programs/Source/` folder and compiles them for use.
Additionally, the script copies the input files from `data/Player-Data/` to the MP-SPDZ `Player-Data/` folder.




#### 3. Run MP-SPDZ programs
We provide example bash scripts on how to run compiled MP-SPDZ programs in `utils/run_experiments/`. These scripts set common parameters (number of parties, party id, network settings, input files, logging) and can be adapted for your experiments.
The example scripts are in `utils/scripts/run_experiments/`:
- `run_spdz_basic.sh` - runs basic protocols (i.e., `since`, `cos`, `exp`, `inv`, `log`, `sqrt`, `inv_sqrt`, `sin`)
- `run_spdz_noise_sampling.sh` - runs noise sampling protocols (i.e., `laplace_its`, `box_muller`, `skellam`, `dgauss_approx`)
- `run_spdz_clipping.sh` - runs the gradient clipping protocol 

The bash script takes the following arguments:
```bash
./run_spdz_<protocol>.sh -p <party_id> -i <number_of_iterations> -f <output_file_path>
```
- The party id (`-p`) should be set to `0`, `1`, ..., `N-1` for `N` parties.
- The number of iterations (`-i`) specifies how many times to run the protocol.
- The output file path (`-f`) specifies where to write the logs.

**How to run on multiple machines**
To run MP-SPDZ programs on multiple machines, you need to:
1. Set up the same environment on each machine (including MP-SPDZ and dependencies).
2. Ensure network connectivity between machines.
3. Add the IP addresses of all participating machines in an `ip-file.txt` file, one per line.
4. Start the MP-SPDZ parties on each machine with the appropriate party id (`-p` flag), total number of parties (`-N` flag) and `-ip` flag pointing to the `ip-file.txt`. 
    - The bash scripts in `utils/scripts/` assume the `ip-file.txt` is in `Player-Data/` in the MP-SPDZ folder.


#### 4. Utility scripts for parsing, plotting and stats
The folder `utils/parsers` contains python scripts to parse MP-SPDZ logs, generate plots and compute statistics from the results.

Specifically:
- `run_parser_and_stats.sh`: runs the log parser and stats computation for the noise sampling experiments.
- `run_parser_and_stats_basic.sh`: runs the log parser and stats computation for the basic MP-SPDZ programs, i.e., `sin`, `cos`, `exp`, `inv`, `log`, `sqrt`, `inv_sqrt`.

#### Helpful Links for MP-SPDZ
* MP-SPDZ github:
https://github.com/data61/MP-SPDZ

* Documentation:
https://mp-spdz.readthedocs.io/en/latest/

