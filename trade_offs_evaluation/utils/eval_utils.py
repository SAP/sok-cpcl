"""
Evaluation Utilities for DP Training Experiments

Provides configuration, timing tracking, and communication accounting utilities
for benchmarking differentially private training.
"""

import json
import pickle
import crypten.communicator
import torch
import crypten
from time import time
from crypten.config import cfg




zeroed_comm_stats = {
    "rounds": 0,
    "bytes": 0,
    "time": 0,
}


non_train_communication_keys = [
    "experiment_name", 
    "train", 
    "per_sample_test", 
    "test", 
    "batch"
]

class EvalConfig():
    """
    Configuration holder for DP training experiments.
    
    Centralizes hyperparameters and settings for reproducibility:
    - Learning rate and optimization parameters
    - Privacy budget (epsilon, delta) and clipping
    - Training hyperparameters (epochs, batch size)
    - Dataset-specific parameters (num_labels, vocab_size)
    - Sampling strategy (sample_rate for Poisson subsampling)
    
    Attributes:
        lr (float): Learning rate
        epsilon (float): Privacy budget epsilon
        delta (float): Privacy failure probability
        clipping_threshold (float): L2 norm gradient clipping bound
        smoothing_factor (float): Label smoothing factor
        per_class_samples (int): Samples per class for balanced batches
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        num_labels (int): Number of classification labels
        vocab_size (int): Vocabulary size (for NLP tasks)
        sample_rate (float): Poisson sampling rate [0,1]
    """
    def __init__(
        self,
        lr=0.001,
        epsilon=1.0,
        delta=1e-5,
        clipping_threshold=10.0,
        smoothing_factor=1e-3,
        per_class_samples=1000,
        num_epochs=2,
        batch_size=10,
        num_labels=2,
        vocab_size=30522,
        softmax_act="softmax",
        hidden_act="relu",
        classifier_act="relu",
        layer_norm_eps=1e-5,
        sample_rate=1.0
    ):
        self.lr = lr
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold
        self.smoothing_factor = smoothing_factor
        self.per_class_samples = per_class_samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.softmax_act = softmax_act
        self.hidden_act = hidden_act
        self.classifier_act = classifier_act
        self.layer_norm_eps = layer_norm_eps
        self.sample_rate = sample_rate

    def __str__(self) -> str:
        return f"EvalConfig(lr={self.lr}, epsilon={self.epsilon}, delta={self.delta}, clipping_threshold={self.clipping_threshold}, smoothing_factor={self.smoothing_factor}, per_class_samples={self.per_class_samples}, num_epochs={self.num_epochs}, batch_size={self.batch_size}, num_labels={self.num_labels}, vocab_size={self.vocab_size}, softmax_act={self.softmax_act}, hidden_act={self.hidden_act}, classifier_act={self.classifier_act}, layer_norm_eps={self.layer_norm_eps}, sample_rate={self.sample_rate})"


class Timing():
    """
    Timing tracker for DP training operations.
    
    Tracks cumulative time for each training phase:
    - forward: Forward pass time
    - backward: Backward pass (gradient computation) time
    - clip: Gradient clipping time
    - noise_sample: DP noise sampling time
    - param_update: Parameter update time
    - epoch: Total epoch time
    - batch: Per-batch training time
    - loss: Loss computation time
    
    Use reset() to clear counters and accumulate() to add new measurements.
    
    Args:
        experiment_name (str): Name for identification/logging
    """
    def __init__(
        self,
        experiment_name 
    ):
        
        self.experiment_name = experiment_name
        self.forward = 0.0
        self.backward = 0.0
        self.clip = 0.0
        self.noise_sample = 0.0
        self.param_update = 0.0
        self.epoch = 0.0
        self.batch = 0.0
        self.loss = 0.0
        self.train = 0.0
        self.pert_param = 0.0
        self.test = 0.0
        self.per_sample_test = 0.0
        self.subsampling = 0.0
        self.grad_accum = 0.0
        self.validation = 0.0

    def reset(self):
        for key in self.__dict__:
            if key != "experiment_name":
                self[key] = 0.0

    def update(self, key, value):
        self[key] = value

    def add(self, key, value):
        self[key] += value

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def __str__(self) -> str:
        return f"Timing for {self.experiment_name}:\n\tforward: {self.forward}\n\tbackward: {self.backward}\n\tclip: {self.clip}\n\tgrad_accu: {self.grad_accum}\n\tnoise_sample: {self.noise_sample}\n\tparam_update: {self.param_update}\n\tepoch: {self.epoch}\n\tbatch: {self.batch}\n\tloss: {self.loss}\n\ttrain: {self.train}\n\tpert_param: {self.pert_param}\n\tsubsampling: {self.subsampling}\n\t\tvalidation: {self.validation}\n\n\ttest: {self.test}\n\t per_sample_test: {self.per_sample_test}\n"
    
    def __repr__(self) -> str:
        return f"Timing for {self.experiment_name}:\n\tforward: {self.forward}\n\tbackward: {self.backward}\n\tclip: {self.clip}\n\tgrad_accu: {self.grad_accum}\n\tnoise_sample: {self.noise_sample}\n\tparam_update: {self.param_update}\n\tepoch: {self.epoch}\n\tbatch: {self.batch}\n\tloss: {self.loss}\n\ttrain: {self.train}\n\tpert_param: {self.pert_param}\n\tsubsampling: {self.subsampling}\n\t\tvalidation: {self.validation}\n\n\ttest: {self.test}\n\t per_sample_test: {self.per_sample_test}\n"
    
    # json serialization
    def to_json(self):
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        return self.__dict__
    
    def compute_per_epochs_timing(self, num_epochs=1, num_batches=1):
        for key in self.__dict__:
            if key not in ["experiment_name", "train", "per_sample_test", "test", "validation", "batch"]:
                self[key] = self[key] / num_epochs
            if key == "batch":
                self["batch"] = self["batch"] / (num_epochs * num_batches)
    
    def compute_per_sample_test_timing(self, num_samples=1):
        self.per_sample_test = self.test / num_samples
        

class Communication():
    def __init__(
        self,
        experiment_name=""
    ):
        # Each communication stats 
        self.experiment_name = experiment_name
        self.forward = zeroed_comm_stats.copy()
        self.backward = zeroed_comm_stats.copy()
        self.clip = zeroed_comm_stats.copy()
        self.noise_sample = zeroed_comm_stats.copy()
        self.param_update = zeroed_comm_stats.copy()
        self.epoch = zeroed_comm_stats.copy()
        self.batch = zeroed_comm_stats.copy()
        self.loss = zeroed_comm_stats.copy()
        self.train = zeroed_comm_stats.copy()
        self.pert_param = zeroed_comm_stats.copy()
        self.test = zeroed_comm_stats.copy()
        self.per_sample_test = zeroed_comm_stats.copy()
        self.subsampling = zeroed_comm_stats.copy()
        self.grad_accum = zeroed_comm_stats.copy()

    def reset(self):
        for key in self.__dict__:
            if key != "experiment_name":
                self[key] = {
                    "rounds": 0,
                    "bytes": 0,
                    "time": 0
                }

    def update(self, key, value):
        self[key] = value

    def add(self, key, value):
        self[key]["rounds"] += value["rounds"]
        self[key]["bytes"] += value["bytes"]
        self[key]["time"] += value["time"]

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)
    
    def __str__(self) -> str:
        return f"Communication for {self.experiment_name}:\n\tForward: {self.forward}\n\tBackward: {self.backward}\n\tClip: {self.clip}\n\tGrad Accu: {self.grad_accum}\n\tNoise Sample: {self.noise_sample}\n\tParam Update: {self.param_update}\n\tEpoch: {self.epoch}\n\tBatch: {self.batch}\n\tLoss: {self.loss}\n\tTrain: {self.train}\n\tPert Param: {self.pert_param}\n\tSubsampling: {self.subsampling}\n\n\tTest: {self.test}\n\tPer Sample Test: {self.per_sample_test}\n"
    
    def __repr__(self) -> str:
        return f"Communication for {self.experiment_name}:\n\tForward: {self.forward}\n\tBackward: {self.backward}\n\tClip: {self.clip}\n\tGrad Accu: {self.grad_accum}\n\tNoise Sample: {self.noise_sample}\n\tParam Update: {self.param_update}\n\tEpoch: {self.epoch}\n\tBatch: {self.batch}\n\tLoss: {self.loss}\n\tTrain: {self.train}\n\tPert Param: {self.pert_param}\n\tSubsampling: {self.subsampling}\n\n\tTest: {self.test}\n\tPer Sample Test: {self.per_sample_test}\n"
    
    # json serialization
    def to_json(self):
        return json.dumps(self.__dict__)
    
    def to_dict(self):
        return self.__dict__
    
    def compute_per_epochs_communication(self, num_epochs=1, num_batches = 1):
        # train communication stats are the sum of all the communication stats
        self.train["rounds"] = sum([self[key]["rounds"] for key in self.__dict__ if key not in non_train_communication_keys])
        self.train["bytes"] = sum([self[key]["bytes"] for key in self.__dict__ if key not in non_train_communication_keys])
        self.train["time"] = sum([self[key]["time"] for key in self.__dict__ if key not in non_train_communication_keys])
        for key in self.__dict__:
            if key != "experiment_name" and key != "train":
                self[key]["rounds"] = self[key]["rounds"] / num_epochs
                self[key]["bytes"] = self[key]["bytes"] / num_epochs
                self[key]["time"] = self[key]["time"] / num_epochs
            if key == "batch":
                self["batch"]["rounds"] = self["batch"]["rounds"] / (num_epochs * num_batches)
                self["batch"]["bytes"] = self["batch"]["bytes"] / (num_epochs * num_batches)
                self["batch"]["time"] = self["batch"]["time"] / (num_epochs * num_batches)


