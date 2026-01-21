import argparse
import logging
import os


parser = argparse.ArgumentParser(description="CNN evaluation")

parser.add_argument(
    "--data_path",
    type=str,
    default="aws-launcher-tmp",
    help="Path to the data directory",
)

parser.add_argument(
    "--noise_type",
    default="local",
    type=str,
    help="Type of noise to evaluate",
    choices=["local", "global"]
)

def _run_experiment(args):
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )

    from trade_offs_evaluation.performance_eval.mpc_noise import run_mpc_noise

    run_mpc_noise(
        args.data_path,
        args.noise_type
    )

def main(run_experiment):
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main(_run_experiment)