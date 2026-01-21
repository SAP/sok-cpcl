import argparse
import logging
import os
try:
    from models import CryptenLeafCNN, CryptenThreeLayerNN
except ImportError:
    from utils.models import CryptenLeafCNN, CryptenThreeLayerNN


parser = argparse.ArgumentParser(description="CNN evaluation")
parser.add_argument(
    "--data_path",
    type=str,
    default="aws-launcher-tmp",
    help="Path to the data directory",
)
parser.add_argument(
    "--download_data",
    default=False,
    action="store_true",
    help="Download the dataset if not in the data_path",
)

parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "--epochs", default=1, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--examples", default=500, type=int, metavar="N", help="number of examples per epoch"
)
parser.add_argument(
    "--lr", "--learning-rate", default=0.1, type=float, help="initial learning rate"
)
parser.add_argument(
    "-eps", "--epsilon", default=10, type=float, help="epsilon for differential privacy"
)
parser.add_argument(
    "--clip_threshold", default=4.0, type=float, help="clip threshold for DP"
)
parser.add_argument(
    "--batch_size", default=500, type=int, help="batch size for training"
)
parser.add_argument(
    "--device", default="cpu", type=str, help="device to run on"
)

parser.add_argument(
    "--n_iterations", default=1, type=int, help="number of iterations to run"
)

parser.add_argument(
    "--model", default="nn", type=str, help="model to run"
)

parser.add_argument(
    "--num_labels", default=10, type=int, help="number of labels"
)

parser.add_argument(
    "--noise_type", default="both", type=str, help="type of noise"
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

    # parse the arguments
    args = parser.parse_args()

    from trade_offs_evaluation.performance_eval.ol_evaluation import run_ol_server

    for non_dp in [True]: #[False, True]:
        batched_options = [True, False] if non_dp else [False]
        for batched in batched_options:
            run_ol_server(
                data_path=args.data_path,
                batch_size=args.batch_size,
                clipping_threshold=args.clip_threshold,
                n_epochs=args.epochs,
                non_dp=non_dp,
                batched=batched,
                num_labels=args.num_labels,
                n_iterations=args.n_iterations,
                noise_type=args.noise_type,
                n_parties=args.world_size,
                n_samples=args.examples,
                model= CryptenThreeLayerNN() if args.model == "nn" else CryptenLeafCNN(),
            )

def main(run_experiment):
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main(_run_experiment)