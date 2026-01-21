#from federated_emnist import get_train_data_emnist, get_test_data_emnist
#from dpsgd_trainer import DP_TrainerEncrypted, DP_Trainer

#from model import Net, CryptenNet
import torch
import logging
import json
import sys
import os
import crypten
import crypten.dp as dp
import numpy as np
import scipy.stats as st
from time import time

try:
    from models import CryptenThreeLayerNN, ThreeLayerNN
except ImportError:
    from utils.models import CryptenThreeLayerNN, ThreeLayerNN

import warnings
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from training import dpsgd_training

import crypten.communicator as comm

from crypten.config import cfg

import crypten.dp as dp 

def calculate_confidence_interval(data, confidence=0.95):
    if len(data) > 1:
        mean = np.mean(data)
        std_err = st.sem(data)
        h = std_err * st.t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean, h
    else:
        return np.mean(data), float('nan')


def run_ol_server(
    data_path, 
    n_epochs=1, 
    non_dp=False, 
    batched=False, 
    model=None,
    num_labels=10, 
    device="cpu", 
    batch_size=20, 
    n_samples=50, 
    n_iterations=1, 
    clipping_threshold=0.05,
    noise_type="local", # It could be "local", "global" or "both"
    n_parties=2
):
    if model is None:
        model = CryptenThreeLayerNN()
    home_path=os.path.expanduser("~")

    try:
        crypten.init(config_file=f"{home_path}/{data_path}/eval_config.yaml")
    except Exception as e:
        print(f"Error: {e}")
        crypten.init()


    x_train_toy = torch.rand(n_samples, 1, 28, 28)
    y_train_toy = torch.randint(0, num_labels, (n_samples,))
    y_train_onehot_toy = torch.nn.functional.one_hot(y_train_toy, num_classes=num_labels)
    lr = 0.15
    delta = 1e-6
    eps = 10

    #home_path=os.path.expanduser("~")
    results = {}
    noise_type_tmp = "local" if noise_type == "both" else noise_type

    results = {
        "n_samples": n_samples,
        "batch_size": batch_size,
        "eps": eps,
        "lr": lr,
        "clipping_threshold": clipping_threshold,
        "delta": delta,
        "n_epochs": n_epochs,
        "n_iterations": n_iterations,
        "dp": str(not non_dp),
        "batched": str(batched),
        "noise_type" : noise_type,
    }

    for i in range(n_iterations):
        logging.info(f"Running non-dp on toy data, iteration {i}")
        print(f"\n\nRunning non-dp on toy data, iteration {i}")
        global_noise_time = 0

        


        exp_iterations = 12 # Standard value is 8 
        with cfg.temp_override({"functions.exp_iterations": exp_iterations}):
            run = dpsgd_training(
                x_train=x_train_toy,
                y_train=y_train_onehot_toy,
                noise_type=noise_type_tmp,
                batch_size=batch_size,
                epochs=n_epochs,
                lr=lr,
                eps=eps,
                clipping_threshold=clipping_threshold,
                delta=delta,
                batched=batched,
                non_dp=non_dp,
                optimizer_type="sgd" if non_dp else "dpsgd",
                num_labels=num_labels,
                model=model,
                device=device,
                n_parties=n_parties,
            )

        #time_grad_compute = run['timing']['forward'] + run['timing']['backward'] + run['timing']['loss'] + run['timing']['clip']
        time_clip = run['timing']['clip']
        time_perturb = run['timing']['noise_sample']
        time_forward = run['timing']['forward']
        time_backward = run['timing']['backward']
        time_loss = run['timing']['loss']
        time_param_update = run['timing']['param_update']
        time_grad_compute = time_forward + time_backward + time_loss + time_clip
        time_total = time_grad_compute + time_perturb + time_param_update

        if noise_type == "both":
            with crypten.no_grad():
                for param in model.parameters():
                    tic = time()
                    dp_noise = dp.dp_utils.sample_global_noise(
                        shape = param.shape,
                        global_std = 5.45,
                        noise_mechanism="gaussian"
                    )
                    param += dp_noise
                    toc = time()
                    global_noise_time += toc - tic
        
        

        # Evaluate also the runtime to send the model to one server to another (to simulate the communication time in FL)
        if n_parties > 1:
            # Here create a tensor that contains all the model parameters
            # Then, measure the time to send this tensor from one party to another
            # This is to simulate the communication time in FL
            model_clear = ThreeLayerNN()
            with crypten.no_grad():
                gradients = [torch.rand_like(param) for param in model_clear.parameters()]
                gradients = [grad.to(device) for grad in gradients]
                comm_0 = comm.get().get_communication_stats()
                gradients_enc = [crypten.cryptensor(grad) for grad in gradients]
                gradients_plain = [grad.get_plain_text() for grad in gradients_enc]
                comm_1 = comm.get().get_communication_stats()
                time_to_send_gradients_fl = comm_1["time"] - comm_0["time"]
                comm_to_send_gradients_fl = comm_1["bytes"] - comm_0["bytes"]
                    

        results[i] = {
            "n_parties": n_parties,
            "time_grad_compute": time_grad_compute,
            "time_clip": time_clip,
            "time_perturb": time_perturb,
            "global_noise_time": global_noise_time,
            "time_to_send_gradients_fl": time_to_send_gradients_fl, # if n_parties > 1 else 0,
            "comm_to_send_gradients_fl": comm_to_send_gradients_fl if n_parties > 1 else 0,
            "time_forward": time_forward,
            "time_backward": time_backward,
            "time_loss": time_loss,
            "time_param_update": time_param_update,
            "time_total": time_total,
        }
    
    # Calculate and store the average times and confidence intervals
    metrics = ["time_grad_compute", "time_clip", "time_perturb", "global_noise_time", "time_to_send_gradients_fl", "time_forward", "time_backward", "time_loss", "time_param_update", "time_total"]
    for metric in metrics:
        if n_parties == 1 and metric == "time_to_send_gradients_fl":
            continue
        data = [results[i][metric] for i in range(n_iterations)]
        mean, confidence_interval = calculate_confidence_interval(data)
        results[metric] = {
            "mean": mean,
            "confidence_interval": confidence_interval,
        }

    # Save results to JSON file
    home_path = os.path.expanduser("~")
    with open(f"{home_path}/{data_path}/results_ol_server{'_batched' if batched else ''}{'' if non_dp else '_dp'}.json", "w") as f:
        json.dump(results, f, indent=4)


        



if __name__ == "__main__":
    n_samples = 500
    batch_size = 500
    n_epochs = 1
    clipping_threshold = 4.0
    n_iterations = 10
    model = CryptenThreeLayerNN()
    num_labels = 10
    device = "cpu"
    noise_type = "both"
    n_parties = 1

    for non_dp in [False, True]:
        batched_options = [True, False] if non_dp else [False]
        for batched in batched_options:
            run_ol_server(
                data_path="soc-cpcl/trade_offs_evaluation/performance_eval/data",
                batch_size=batch_size,
                clipping_threshold=clipping_threshold,
                n_epochs=n_epochs,
                non_dp=non_dp,
                batched=batched,
                num_labels=num_labels,
                n_iterations=n_iterations,
                noise_type=noise_type,
                n_parties=n_parties,
                n_samples=n_samples,
                model= model,
            )
    
