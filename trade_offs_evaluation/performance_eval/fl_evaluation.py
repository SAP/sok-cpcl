"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

import torch
import logging
import json
import sys
import os
import crypten
import numpy as np
import scipy.stats as st
from time import time
from training import clear_training
try:
    from models import ThreeLayerNN
except ImportError:
    from utils.models import ThreeLayerNN
import warnings

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def encrypt_gradients(model, device):
    crypten.init()
    model = model.to(device)
    gradients = [torch.rand_like(param) for param in model.parameters()]
    gradients = [grad.to(device) for grad in gradients]
    tic = time()
    gradients_enc = [crypten.cryptensor(grad) for grad in gradients]
    toc = time()
    encrypt_grad_time = toc - tic
    return gradients_enc, encrypt_grad_time

def aggregate_gradients(gradients_enc, n_clients=100):
    gradients_enc_all_clients = [gradients_enc for _ in range(n_clients)]
    tic = time()
    stacked_gradients = [crypten.stack([grad[i] for grad in gradients_enc_all_clients]) for i in range(len(gradients_enc))]
    sum_grad_all_clients = [stacked_gradients[i].sum(dim=0) for i in range(len(stacked_gradients))]
    toc = time()
    aggregation_time = toc - tic
    return aggregation_time

def calculate_confidence_interval(data, confidence=0.95):
    if len(data) > 1:
        mean = np.mean(data)
        std_err = st.sem(data)
        h = std_err * st.t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean, h
    else:
        warnings.warn("Sample size is too small; returning NaN for confidence interval.")
        return np.mean(data), float('nan')

def run_fl_client(
    data_path, 
    n_epochs=1, 
    non_dp=False, 
    batched=False, 
    model=ThreeLayerNN(), 
    num_labels=10, 
    device="cpu", 
    batch_size=20, 
    n_samples=50, 
    n_iterations=1, 
    clipping_threshold=0.05
):
    eps = 10
    lr = 0.1
    delta = 1e-6
    x_train = torch.rand(n_samples, 1, 28, 28)
    y_train = torch.randint(0, num_labels, (n_samples,))

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
    }

    for i in range(n_iterations):
        print("Running iteration", i)
        dp = clear_training(
            x_train=x_train,
            y_train=y_train,
            validate=False,
            test=False,
            lr=lr,
            eps=eps,
            clipping_threshold=clipping_threshold,
            delta=delta,
            batch_size=batch_size,
            epochs=n_epochs,
            experiment_name="dp" if not non_dp else "non_dp",
            non_dp=non_dp,
            batched=batched,
            sampling_rate=batch_size / n_samples,
            model=model,
            num_labels=num_labels,
            device=device,
            optimizer_type="dpsgd" if not non_dp else "sgd_clear",
        )

        gradients_enc, encrypt_grad_time = encrypt_gradients(model, device)
        aggregation_time = aggregate_gradients(gradients_enc)

        time_clip = dp['timing']['clip']
        time_perturb = dp['timing']['noise_sample']
        time_param_update = dp['timing']['param_update']
        time_forward = dp['timing']['forward']
        time_loss = dp['timing']['loss']
        time_backward = dp['timing']['backward']
        time_grad_compute = time_forward + time_backward + time_loss + time_clip
        total_time = time_grad_compute + time_perturb + time_param_update

        results[i] = {
            "time_grad_compute": time_grad_compute,
            "time_clip": time_clip,
            "time_perturb": time_perturb,
            "time_encrypt_grad": encrypt_grad_time,
            "time_aggregation": aggregation_time,
            "time_param_update": time_param_update,
            "time_forward": time_forward,
            "time_backward": time_backward,
            "time_loss": time_loss,
            "time_total": total_time
        }

    # Calculate and store the average times and confidence intervals
    metrics = ["time_perturb", "time_clip", "time_grad_compute", "time_encrypt_grad", "time_aggregation", "time_param_update", "time_forward", "time_backward", "time_loss", "time_total"]
    for metric in metrics:
        data = [results[i][metric] for i in range(n_iterations)]
        mean, confidence_interval = calculate_confidence_interval(data)
        results[metric] = {
            "mean": mean,
            "confidence_interval": confidence_interval,
        }

    # Save results to JSON file
    home_path = os.path.expanduser("~")
    with open(f"{home_path}/{data_path}/results_fl_client{'_batched' if batched else ''}{'' if non_dp else '_dp'}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    n_samples = 50
    batch_size = 10
    n_epochs = 1
    clipping_threshold = 0.05
    n_iterations = 10
    model = ThreeLayerNN()
    num_labels = 10
    device = "cpu"

    for non_dp in [False, True]:
        batched_options = [True, False] if non_dp else [False]
        for batched in batched_options:
            run_fl_client(
                data_path="soc-cpcl/trade_offs_evaluation/performance_eval/data",
                batch_size=batch_size,
                clipping_threshold=clipping_threshold,
                n_epochs=n_epochs,
                non_dp=non_dp,
                batched=batched,
                num_labels=num_labels,
                n_iterations=n_iterations,
                n_samples=n_samples,
            )