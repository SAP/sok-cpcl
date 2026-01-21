"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

try: 
    from models import CryptenThreeLayerNN
except ImportError:
    from utils.models import CryptenThreeLayerNN

import crypten
import torch
import logging
import os

import crypten.dp as dp 

import crypten.communicator as comm
from time import time
import sys
import json

logging.basicConfig(level=logging.INFO, stream=sys.stdout)




def run_mpc_noise(
    data_path = None, 
    noise_type = "local"
):
    home_path=os.path.expanduser("~")
    crypten.init(config_file=f"{home_path}/{data_path}/eval_config.yaml")

    model = CryptenThreeLayerNN()
    model.encrypt()

    noise_stddev = 2.83

    stats = {}

    for i in range(10):
        logging.info(f"Running MPC noise on toy data, iteration {i}")
        comm.get().reset_communication_stats()
        logging.info(f"Reset communication stats {comm.get().get_communication_stats()}")
        noise_time = 0
        for param in model.parameters():
            tic = time()
            logging.info(f"Generating noise for param {param.shape}")
            if noise_type == "global":
                dp_noise = dp.dp_utils.sample_global_noise(
                    shape = param.shape,
                    global_std = noise_stddev,
                    noise_mechanism="gaussian"
                )
            else:
                dp_noise = dp.dp_utils.sample_local_noise(
                shape = param.shape, 
                local_std=noise_stddev, 
                noise_mechanisms="gaussian"
            )
            toc = time()
            noise_time += toc - tic
            logging.info(f"Time to generate noise: {toc-tic}")
        
        comm_stats = comm.get().get_communication_stats()

        stats[i] = {
            "noise_time": noise_time,
            "comm_stats": comm_stats
        }

        with open(f"{home_path}/{data_path}/noise_stats_{noise_type}_{i}.json", 'w') as f:
            json.dump(stats, f)
    
    with open(f"{home_path}/noise_stats_{noise_type}.json", 'w') as f:
        json.dump(stats, f)
    



if __name__ == "__main__":
    run_mpc_noise(
        data_path=f"sok-cpcl/trade_offs_evaluation/performance_eval/data",
        noise_type="local"
    )