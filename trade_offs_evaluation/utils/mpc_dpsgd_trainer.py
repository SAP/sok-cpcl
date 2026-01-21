"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

"""
Differentially Private Encrypted Trainer

This module provides DP_TrainerEncrypted, a trainer class for training models
on encrypted data (using CrypTen) with differentially private stochastic gradient descent (DP-SGD).

Supports:
- Local and global DP noise 
- Privacy accounting via Renyi Differential Privacy (RDP)
- Multi-party encrypted training via CrypTen
- Per-example gradient clipping 
"""

import torch
import torch.nn as nn
import crypten
import crypten.nn as cnn
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import math
from crypten.dp.dp_utils import l2_norm_clip, compute_stddev_from_epsilon, sample_local_noise, l2_norm_clip_optim
from eval_utils import Timing
from time import time
import logging

optimizers = {
    "dpsgd": crypten.optim.DPSGD,
    "sgd": crypten.optim.SGD,
    "sgd_clear": torch.optim.SGD
}

class DP_TrainerEncrypted():
    """
    Trainer for differentially private encrypted model training.
    
    This class wraps a CrypTen encrypted model and provides training with
    differential privacy guarantees. It handles:
    - Gradient clipping to bound sensitivity
    - DP noise sampling (local or global)
    - Privacy budget accounting
    - Multi-party secure computation
    
    Args:
        model (crypten.nn.Module): Encrypted or plain model to train
        batch_size (int): Batch size for training
        num_labels (int): Number of classification labels
        num_epochs (int): Number of training epochs
        loss_fn: Loss function (default: CrossEntropyLoss)
        optimizer_type (str): Optimizer name ("dpsgd", "sgd", "sgd_clear")
        noise_mechanism (str): DP noise type ("gaussian", "skellam", "poisson-binomial")
        noise_type (str): "local" (per-party) or "global" (joint) DP
        noise_multiplier (float): Multiplier for noise std dev (overrides epsilon-based)
        epsilon (float): Privacy budget epsilon (used if noise_multiplier is None)
        delta (float): Privacy failure probability delta
        clipping_threshold (float): L2 norm clipping threshold for gradients
        lr (float): Learning rate
        n_parties (int): Number of parties in MPC (for local DP scaling)
        sampling_rate (float): Poisson sampling rate [0,1]
        timing (Timing): Optional timing tracker object
        verbose (bool): Print debug information
        collusion_multiplier (float): Multiplier to simulate collusion (1.0 = no collusion)
        experiment_name (str): Name for logging/identification
        device (str): "cpu" or "cuda"
        subsampling_type (str): "approx" or "poisson" for subsampling
    """
    def __init__(
        self, 
        model,
        batch_size = 1, 
        num_labels = 2, 
        num_epochs = 10,
        loss_fn = cnn.CrossEntropyLoss(),
        optimizer_type = "dpsgd",
        noise_mechanism = "gaussian",
        noise_type = "local",
        noise_multiplier = None,
        epsilon = 1.0,
        delta = 1e-5,
        clipping_threshold = 50.0,
        lr = 0.0005,
        n_parties = 1,
        sampling_rate = 1.0,
        timing = None,
        verbose = False,
        collusion_multiplier = 1,
        experiment_name = "dpsgd",
        device = "cpu",
        subsampling_type = "approx"
    ):
        self.model = model.to(device)

        self.device = device
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold
        self.sample_rate = sampling_rate
        self.subsampling_type = subsampling_type
        self.verbose = verbose


        if timing is None:
            self.timing = Timing(experiment_name=experiment_name)
        else:
            self.timing = timing
        

        if noise_multiplier is None:
            # The std dev if local should be divided by the number of parties
            print(f"Noise multiplier not define, computing noise std dev with RDP")
            self.noise_stddev = compute_stddev_from_epsilon(
                noise_mechanism = noise_mechanism,
                noise_type = noise_type,
                eps = epsilon,
                delta = delta,
                clipping_threshold = clipping_threshold,
                num_parties = n_parties,
                epochs = num_epochs,
                sampling_rate = sampling_rate
            )
        else:
            self.noise_stddev = noise_multiplier * clipping_threshold
            if noise_type == "local":
                self.noise_stddev /= n_parties


        # To simulate collusion, we can multiply the noise by a factor
        self.noise_stddev *= collusion_multiplier
        logging.debug(f"Noise stddev: {self.noise_stddev}")
        
        self.optimizer_type = optimizer_type
        
        try:
            if "dp" in optimizer_type:
                # Additional parameters for the DPSGD optimizer
                self.optimizer = optimizers[optimizer_type](
                    params=self.model.parameters(),
                    lr=lr,
                    l2_clipping_threshold=clipping_threshold,
                    noise_stddev=self.noise_stddev,
                    noise_mechanism=noise_mechanism,
                    noise_type=noise_type,
                    device=device
                )
            else:
                self.optimizer = optimizers[optimizer_type](
                    params=self.model.parameters(),
                    lr=lr
            )    
        except KeyError:
                raise ValueError(f"Invalid optimizer type: {optimizer_type}")
        
        self.trainable_params = { name : param for name, param in self.model.named_parameters() if param.requires_grad}
        
    def reset_timing(self):
        self.timing.reset()


    def get_timing(self):
        return self.timing

    def l2_norm_clip_optim(self, x, clip_threshold):
        """ This should me an MPC function """
        squared_x_sum = x.square().sum()
        # piecewise inv sqrt approximation
        threshold =1e2
        scale =1e3
        eps = 1e-5
        scaling_cond = squared_x_sum.le(threshold)
        squared_x_sum = squared_x_sum + eps
        scaled_var = squared_x_sum/scale

        inv_sqrt_arg = scaled_var + (squared_x_sum - scaled_var)*(scaling_cond)
        inv_sqrt_out = inv_sqrt_arg.inv_sqrt()
        inv_l2_norm = inv_sqrt_out + (inv_sqrt_out*(1/(math.sqrt(scale))) - inv_sqrt_out)*(1-scaling_cond)
        
        #inv_l2_norm = squared_x_sum.inv_sqrt()
        tmp = inv_l2_norm * clip_threshold
        clip = (tmp).lt(1)
        x = (x * tmp - x) * clip + x

        return x
        #return (x * tmp - x) * clip + x



    def compute_clip_gradient(self, loss_value):
        tic = time()
        loss_value.backward()
        toc = time()
        self.timing.add("backward", toc - tic)
        for param in self.trainable_params.values():
            # Added for compatibility with LoRA, since not all the parameters are trainable
            #if param.requires_grad:    
            per_sample_grad = param.grad.detach().clone()
            
            tic_clip = time()
            #per_sample_grad = l2_norm_clip(per_sample_grad, clip_threshold=self.clipping_threshold)
            per_sample_grad = self.l2_norm_clip_optim(per_sample_grad, clip_threshold=self.clipping_threshold)
            #self.l2_norm_clip_optim(per_sample_grad, clip_threshold=self.clipping_threshold)

            toc_clip = time()
            self.timing.add("clip", toc_clip - tic_clip)

            param.accumulated_grads += per_sample_grad

        
        if self.verbose:
            loss_plain = loss_value.get_plain_text() if isinstance(loss_value, crypten.CrypTensor) else loss_value
            logging.info(f"\t\tSample loss value: {loss_plain}")            
            print(f"\t\tSample loss value: {loss_plain}") 

    
    def optimizer_step(self, batch_size):
        # Update the model parameters
        #for param in self.model.parameters():
        tic = time()
        for param in self.trainable_params.values():
            param.grad = param.accumulated_grads
        if "dp" in self.optimizer_type:
            self.optimizer.step(batch_size)
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        toc = time()
        if "dp" in self.optimizer_type:
            self.timing.add("noise_sample", self.optimizer.get_noise_time())
            self.optimizer.reset_noise_time()
        self.timing.add("param_update", toc - tic)



    

    def subsample_data(
        self, 
        x, 
        y,
    ):
        tic = time()
        n_samples = x.size(0)
        
        if self.subsampling_type == "approx":
            n_subsamples = math.ceil(n_samples * self.sample_rate)
            
            # To share the same random seed
            idx = torch.randperm(n_samples)
            idx_enc =  crypten.cryptensor(idx, device=self.device)
            idx = idx_enc.get_plain_text()
            idx = idx.to(torch.long)

            x_sub = x[idx][:n_subsamples]   
            y_sub = y[idx][:n_subsamples]
        elif self.subsampling_type == "poisson":
            # one side sampling for computational efficiency (NOT correct for accounting)
            prob = torch.rand(n_samples)

            # Encryption and decryption to share the same values
            prob_enc = crypten.cryptensor(prob, device=self.device)
            prob = prob_enc.get_plain_text()

            mask = (prob < self.sample_rate)

            x_sub = x[mask]
            y_sub = y[mask]
            #samples_rates = torch.tensor([self.sample_rate] * n_samples)

            # To share the same mask
            #poisson_samples = torch.poisson(samples_rates)
            #poisson_samples_enc = crypten.cryptensor(poisson_samples, device=self.device)
            #poisson_samples = poisson_samples_enc.get_plain_text()

            #mask = poisson_samples > 0
            #x_sub = x[mask]
            #y_sub = y[mask]
        else:
            raise ValueError(f"Invalid subsampling type: {type}")
        
        toc = time()
        self.timing.add("subsampling", toc - tic)
        return x_sub, y_sub


    def train_epoch_batched(
        self,
        x,
        y
    ):
        dataset_size = x.size(0)
        epoch_loss = 0
        steps = 0
        total_steps = math.ceil(dataset_size / self.batch_size)
        for _ in range(0, dataset_size, self.batch_size):
            steps += 1
            # the number of iterations for epochs is  = dataset_size / batch_size
            x_sub, y_sub = self.subsample_data(x, y)
            #x_sub,y_sub = x[:self.batch_size], y[:self.batch_size]
            logging.info(f"Subsampled data: {x_sub.size(0)}")
            print(f"Subsampled data: {x_sub.size(0)}")


            for param in self.trainable_params.values():
                param.accumulated_grads = 0

            # Forward pass
            tic = time()
            logits = self.model(x_sub)
            toc = time()
            self.timing.add("forward", toc - tic)

            # Compute the loss
            tic = time()
            loss_value = self.loss_fn(logits, y_sub)
            toc = time()
            self.timing.add("loss", toc - tic)

            if "dp" in self.optimizer_type:
                self.compute_clip_gradient(loss_value)
            else:
                    raise ValueError("Only DPSGD is supported for non-batched training")
            with crypten.no_grad():
                samples_loss += loss_value
            
            self.optimizer_step(x_sub.size(0))
            batch_loss = samples_loss.get_plain_text() / x_sub.size(0)
            print(f"\tBatch {steps}/{total_steps} loss value: {batch_loss}")# (cleartext loss value: {samples_loss_plain / x_sub.size(0)})")

            epoch_loss += batch_loss
        return epoch_loss / steps


    def train_epoch(
        self,
        x,
        y
    ):
        dataset_size = x.size(0)
        epoch_loss = 0
        steps = 0
        total_steps = math.ceil(dataset_size / self.batch_size)
        for _ in range(0, dataset_size, self.batch_size):
            steps += 1
            # the number of iterations for epochs is  = dataset_size / batch_size
            x_sub, y_sub = self.subsample_data(x, y)
            #x_sub,y_sub = x[:self.batch_size], y[:self.batch_size]
            logging.info(f"Subsampled data: {x_sub.size(0)}")
            print(f"Subsampled data: {x_sub.size(0)}")

            for param in self.trainable_params.values():
                param.accumulated_grads = 0

            samples_loss = 0
            samples_loss_plain = 0
            for i in range(x_sub.size(0)):
                x_i = x_sub[i].unsqueeze(0).to(self.device)
                y_i = y_sub[i].unsqueeze(0).to(self.device)

                tic = time()
                logits = self.model(x_i)
                logits_plain = logits.get_plain_text() if isinstance(logits, crypten.CrypTensor) else logits
                if self.verbose:
                    print(f"Logits: {logits_plain}")
                if (logits_plain.abs() > 10e5).any():
                    print(f"Overflow in logits: {logits_plain}")
                toc = time()
                self.timing.add("forward", toc - tic)
                
                # To avoid underflow in the loss function
                #logits = logits * 100
                tic = time()
                loss_value = self.loss_fn(logits, y_i)
                toc = time()
                self.timing.add("loss", toc - tic)
                #loss_plain = torch.nn.functional.cross_entropy(logits_plain, y_i.get_plain_text())

                if "dp" in self.optimizer_type:
                    self.compute_clip_gradient(loss_value)
                else:
                    raise ValueError("Only DPSGD is supported for non-batched training")
                with crypten.no_grad():
                    samples_loss += loss_value
                    #samples_loss_plain += loss_plain

            self.optimizer_step(x_sub.size(0))
            
            batch_loss = samples_loss.get_plain_text() / x_sub.size(0)
            print(f"\tBatch {steps}/{total_steps} loss value: {batch_loss}")# (cleartext loss value: {samples_loss_plain / x_sub.size(0)})")

            epoch_loss += batch_loss
        return epoch_loss / steps
    

    def train(
        self,
        x,
        y,
        batched = False
    ):         
        self.model.train()
        self.reset_timing()
        losses = []
        for epoch in range(self.num_epochs):
            tic = time()
            if batched:
                loss_value = self.train_epoch_batched(x, y)
            else:
                loss_value = self.train_epoch(x, y)
            toc = time()
            losses.append(loss_value.get_plain_text() if isinstance(loss_value, crypten.CrypTensor) else loss_value)
            self.timing.add("epoch", toc - tic)
            logging.info(f"Epoch {epoch} Loss {loss_value} Time: {toc-tic}")
            print(f"Epoch {epoch} Loss {loss_value} Time: {toc-tic}")

        self.timing.update("train", self.timing.__getitem__("epoch"))
        self.timing.compute_per_epochs_timing(epoch+1, num_batches=1)

        return losses

    
    def validate(
        self,
        x,
        y,
        y_onehot,
        validation_metric = "accuracy"
    ):
        tic = time()
        self.model.eval()
        n_batches = math.ceil(x.size(0) / self.batch_size)
        y_pred = []
        val_loss = 0
        for i in range(n_batches):
            x_batch = x[i*self.batch_size:(i+1)*self.batch_size].to(self.device)
            y_batch = y_onehot[i*self.batch_size:(i+1)*self.batch_size].to(self.device)

            logits = self.model(x_batch)
            plain_logits = [l.get_plain_text() if isinstance(l, crypten.CrypTensor) else l for l in logits]
            y_pred += [l.argmax().item() for l in plain_logits]


            val_loss += self.loss_fn(logits, y_batch)
        
        val_loss /= n_batches
        val_loss = val_loss.get_plain_text() if isinstance(val_loss, crypten.CrypTensor) else val_loss
        
        y_true = y.get_plain_text() if isinstance(y, crypten.CrypTensor) else y
        if "mps" in self.device:
            y_pred = torch.tensor(y_pred).cpu()
            y_true = y_true.cpu()
        else: 
            y_pred = torch.tensor(y_pred).to(self.device)
            y_true = y_true.to(self.device)

        if validation_metric == "accuracy":
            val_score = accuracy_score(y_true, y_pred)
        elif validation_metric == "f1":
            val_score = f1_score(y_true, y_pred, average="micro")
        elif validation_metric == "mcc":
            val_score =  matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Invalid validation metric: {validation_metric}")
        toc = time()
        self.timing.add("validation", toc - tic)

        return val_score, val_loss
        
    
    def train_and_validate(
        self,
        x,
        y,
        x_val,
        y_val,
        y_val_onehot,
        validation_metric = "accuracy",
        model_name = "best_model.pth",
        batched = False,
        validation_freq = 1
    ):
        self.reset_timing()
        losses = []
        val_scores = []
        val_losses = []
        for epoch in range(self.num_epochs):
            tic = time()
            self.model.train()
            if batched:
                loss = self.train_epoch_batched(x, y)
            else:   
                loss = self.train_epoch(x, y)
            toc = time()
            self.timing.add("epoch", toc - tic)
            plain_loss = loss.get_plain_text() if isinstance(loss, crypten.CrypTensor) else loss
            logging.info(f"Epoch {epoch} Loss {plain_loss} Time: {toc-tic}")           
            print(f"Epoch {epoch} Loss {plain_loss} Time: {toc-tic}")            
            losses.append(plain_loss)
            if plain_loss < 0:
                logging.info("Negative loss value (over/underflow)")
                print("Negative loss value (over/underflow)")
                break

                

            if epoch % validation_freq == 0: 
                tic_val = time()
                val_metric, val_loss = self.validate(
                                            x =x_val,   
                                            y = y_val, 
                                            y_onehot = y_val_onehot,
                                            validation_metric= validation_metric
                                            )
                toc_val = time()
                
                logging.info(f"Validation {validation_metric.capitalize()}: {val_metric} Loss: {val_loss} Time: {toc_val-tic_val}") 
                print(f"Validation {validation_metric.capitalize()}: {val_metric} Time: {toc_val-tic_val}") 
                
                val_scores.append(val_metric)
                val_losses.append(val_loss)
                # Save the model if the validation metric is better
                if epoch > 0: 
                    max_val_score = max(val_scores[:-1])
                    if val_scores[-1] > max_val_score:
                        crypten.save(self.model, f"{model_name}.pth")
                        val_score_plain = val_scores[-1].get_plain_text() if isinstance(val_scores[-1], crypten.CrypTensor) else val_scores[-1]
                        val_score_old = max_val_score.get_plain_text() if isinstance(max_val_score, crypten.CrypTensor) else max_val_score

                        logging.info(f"Validation score improved from {val_score_old} to {val_score_plain}")
                        print(f"Validation score improved from {val_score_old} to {val_score_plain}")
                else:
                    crypten.save(self.model, f"{model_name}.pth")

        self.timing.update("train", self.timing.__getitem__("epoch"))
        self.timing.compute_per_epochs_timing(epoch+1, num_batches=1)

        return losses, val_scores, val_losses



class DP_Trainer(DP_TrainerEncrypted):
    def __init__(
        self, 
        model,
        batch_size = 1, 
        num_labels = 2, 
        num_epochs = 10,
        loss_fn = nn.CrossEntropyLoss(),
        optimizer_type = "dpsgd",
        noise_mechanism = "gaussian",
        noise_type = "local",
        noise_multiplier = None,
        epsilon = 1.0,
        delta = 1e-5,
        clipping_threshold = 50.0,
        lr = 0.0005,
        n_parties = 1,
        sampling_rate = 1.0,
        timing = None,
        verbose = False,
        collusion_multiplier = 1,
        experiment_name = "dpsgd",
        device = "cpu",
        subsampling_type = "approx"
    ):
        self.model = model.to(device)

        self.device = device
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold
        self.sample_rate = sampling_rate
        self.subsampling_type = subsampling_type
        self.verbose = verbose


        if timing is None:
            self.timing = Timing(experiment_name=experiment_name)
        else:
            self.timing = timing
        

        if noise_multiplier is None:
            # The std dev if local should be divided by the number of parties
            print(f"Noise multiplier not defined, computing noise std dev with RDP")
            self.noise_stddev = compute_stddev_from_epsilon(
                noise_mechanism = noise_mechanism,
                noise_type = noise_type,
                eps = epsilon,
                delta = delta,
                clipping_threshold = clipping_threshold,
                num_parties = n_parties,
                epochs = num_epochs,
                sampling_rate = sampling_rate
            )
        else:
            self.noise_stddev = noise_multiplier * clipping_threshold
            if noise_type == "local":
                self.noise_stddev /= n_parties


        # To simulate collusion, we can multiply the noise by a factor
        self.noise_stddev *= collusion_multiplier
        logging.debug(f"Noise stddev: {self.noise_stddev}")
        
        self.optimizer_type = optimizer_type
        if optimizer_type == "dpsgd":
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=lr
            )
        else:
            try:
                self.optimizer = optimizers[optimizer_type](
                    params=self.model.parameters(),
                    lr=lr
                )
            except KeyError:
                raise ValueError(f"Invalid optimizer type: {optimizer_type}")
        
        self.trainable_params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}


    def compute_clip_gradient(self, loss_value):
        tic = time()
        loss_value.backward()
        toc = time()
        self.timing.add("backward", toc - tic)
        #for param in self.model.parameters():
        for param in self.trainable_params.values():
            tic = time()
            per_sample_grad = param.grad.detach().clone()
            norm = per_sample_grad.norm(p=2)

            per_sample_grad = min(1, self.clipping_threshold / norm) * per_sample_grad
            toc = time()
            #div_factor = norm / self.clipping_threshold
            #if div_factor > 1:
            #    per_sample_grad = per_sample_grad / div_factor
            toc = time()
            self.timing.add("clip", toc - tic)
            param.accumulated_grads += per_sample_grad

        if self.verbose:
            logging.info(f"\t\tSample loss value: {loss_value:.4f}")            
            print(f"\t\tSample loss value: {loss_value:.4f}")   

    def optimizer_step(self, batch_size):
        #for param in self.model.parameters():
        tic = time()
        for param in self.trainable_params.values():
            # sample noise
            tic = time()
            #noise = torch.randn_like(param) * self.noise_stddev
            noise = torch.normal(mean=0, std=self.noise_stddev, size=param.size())
            toc = time()
            self.timing.add("noise_sample", toc - tic)
            # add noise to the gradients
            # Need to convert noise to dtype of the gradients (float32)
            param.accumulated_grads += noise.to(dtype=torch.float32).to(self.device)
            # average the gradients over the batch
            param.grad = param.accumulated_grads / batch_size

        self.optimizer.step()
        self.optimizer.zero_grad()   
        toc = time()
        self.timing.add("param_update", toc - tic)      


    def subsample_data(
        self, 
        x, 
        y,
    ):
        tic = time()
        n_samples = x.size(0)
        if self.subsampling_type == "approx":
            n_subsamples = math.ceil(n_samples * self.sample_rate)
            idx = torch.randperm(n_samples)
            x_sub = x[idx][:n_subsamples]   
            y_sub = y[idx][:n_subsamples]
        elif self.subsampling_type == "poisson":
            prob = torch.rand(n_samples)
            mask = (prob < self.sample_rate)
            x_sub = x[mask]
            y_sub = y[mask]

        else:
            raise ValueError(f"Invalid subsampling type: {type}")
        toc = time()
        self.timing.add("subsampling", toc - tic)
        return x_sub, y_sub


    def train_epoch_batched(
        self,
        x,
        y
    ):
        dataset_size = x.size(0)
        epoch_loss = 0
        steps = 0
        total_steps = math.ceil(dataset_size / self.batch_size)
        for _ in range(0, dataset_size, self.batch_size):
            steps += 1
            # the number of iterations for epochs is  = dataset_size / batch_size
            x_sub, y_sub = self.subsample_data(x, y)

            if "dp" not in self.optimizer_type:
                raise ValueError("Only DPSGD is supported for per-example training")

            for p in self.model.parameters():
                p.accumulated_grads = 0  

                tic = time()
                logits = self.model(x_sub.to(self.device))
                toc = time()
                self.timing.add("forward", toc - tic)
                tic = time()
                loss_value = self.loss_fn(logits, y_sub.to(self.device))
                toc = time()
                self.timing.add("loss", toc - tic)
            

            if "dp" in self.optimizer_type:
                self.compute_clip_gradient(loss_value)
            else:
                    raise ValueError("Only DPSGD is supported for non-batched training")
            with crypten.no_grad():
                samples_loss += loss_value

            self.optimizer_step(x_sub.size(0))
            batch_loss = samples_loss.get_plain_text() / x_sub.size(0)
            print(f"\tBatch {steps}/{total_steps} loss value: {batch_loss}")# (cleartext loss value: {samples_loss_plain / x_sub.size(0)})")

            epoch_loss += batch_loss
        return epoch_loss / steps
    


    def train_epoch(self, x, y):
        dataset_size = x.size(0)
        epoch_loss = 0
        steps = 0
        total_steps = math.ceil(dataset_size / self.batch_size)
        for _ in range(0, dataset_size, self.batch_size):
            steps += 1
            # the number of iterations for epochs is  = dataset_size / batch_size
            x_sub, y_sub = self.subsample_data(x, y)

            if "dp" not in self.optimizer_type:
                raise ValueError("Only DPSGD is supported for per-example training")

            for p in self.model.parameters():
                p.accumulated_grads = 0     

            samples_loss = 0

            for i in range(x_sub.size(0)):
                x_i = x_sub[i].unsqueeze(0).to(self.device)
                y_i = y_sub[i].unsqueeze(0).to(self.device)

                tic = time()
                logits = self.model(x_i)
                toc = time()
                self.timing.add("forward", toc - tic)

                tic = time()
                loss_value = self.loss_fn(logits, y_i)
                toc = time()
                self.timing.add("loss", toc - tic)

                self.compute_clip_gradient(loss_value)   
                with torch.no_grad():
                    samples_loss += loss_value
            self.optimizer_step(x_sub.size(0))
            
            batch_loss = samples_loss / x_sub.size(0)
            print(f"\tBatch {steps}/{total_steps} loss value: {batch_loss}")# (cleartext loss value: {samples_loss_plain / x_sub.size(0)})")

            epoch_loss += batch_loss
        return epoch_loss / steps
        

    def train(
        self, 
        x, 
        y,
        batched = False, 
    ):
        self.model.train()
        self.timing.reset()
        losses = []
        for epoch in range(self.num_epochs):
            tic = time()
            if batched:
                loss_value = self.train_epoch_batched(x, y)
            else:
                loss_value = self.train_epoch(x, y)
            toc = time()
            losses.append(loss_value)
            logging.info(f"Clear: Epoch {epoch} Loss {loss_value} Time: {toc-tic}")        
            print(f"Clear: Epoch {epoch} Loss {loss_value} Time: {toc-tic}")        
            self.timing.add("epoch", toc - tic)
            

        self.timing.update("train", self.timing.__getitem__("epoch"))
        self.timing.compute_per_epochs_timing(epoch+1, num_batches=1)

        torch.save(self.model.state_dict(), self.model_path)

        return losses


    def validate(
        self, 
        x, 
        y, 
        validation_metric = "accuracy"
    ):
        tic = time()
        self.model.eval()
        n_batches = math.ceil(x.size(0) / self.batch_size)
        y_true = y
        y_pred = []
        val_loss = 0
        for i in range(n_batches):
            x_batch = x[i*self.batch_size:(i+1)*self.batch_size].to(self.device)
            logits = self.model(x_batch).to(self.device)
            y_pred += [l.argmax().item() for l in logits]
            val_loss += self.loss_fn(logits, y_true[i*self.batch_size:(i+1)*self.batch_size].to(self.device))

        val_loss /= n_batches
        y_pred = torch.tensor(y_pred)
        #if "mps" in self.device:
        #    y_pred = y_pred.cpu()
        #    y_true = y_true.cpu()
        #else:
        #    y_pred = y_pred.to(self.device)
        #    y_true = y_true.to(self.device)

        if validation_metric == "accuracy":
            val_score = accuracy_score(y_true, y_pred)
        elif validation_metric == "f1":
            val_score = f1_score(y_true, y_pred, average="micro")
        elif validation_metric == "mcc":
            val_score = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Invalid validation metric: {validation_metric}")
        toc = time()
        self.timing.add("validation", toc - tic)
        return val_score, val_loss
          

    def train_and_validate(
        self, 
        x, 
        y, 
        x_val, 
        y_val, 
        validation_metric = "accuracy",
        model_name = "best_model_clear",
        batched = False,
        validation_freq = 1
    ):
        self.timing.reset()
        losses = []
        val_scores = []
        val_losses = []
        for epoch in range(self.num_epochs):
            tic = time()
            self.model.train()
            if batched:
                loss = self.train_epoch_batched(x,y)
            else:   
                loss = self.train_epoch(x, y)
            toc = time()
            self.timing.add("epoch", toc - tic)
            logging.info(f"Epoch {epoch} Loss {loss} Time: {toc-tic}")            
            print(f"Epoch {epoch} Loss {loss} Time: {toc-tic}")            
            losses.append(loss)


            if epoch % validation_freq == 0: 
                tic_val = time()
                val_metric, val_loss = self.validate(x_val, y_val, validation_metric)
                toc_val = time()
                
                
                logging.info(f"Validation {validation_metric.capitalize()}: {val_metric} Val Loss: {val_loss} Time: {toc_val-tic_val}") 
                print(f"Validation {validation_metric.capitalize()}: {val_metric} Loss: {val_loss} Time: {toc_val-tic_val}") 
                
                val_scores.append(val_metric)
                val_losses.append(val_loss)
                # Save the model if the validation metric is better
                #if epoch == 0 or val_scores[-1] > max(val_scores[:-1]):
                #    if epoch > 0:
                #        logging.info(f"Validation score improved from {max(val_scores[:-1])} to {val_scores[-1]}")
                #        print(f"Validation score improved from {max(val_scores[:-1])} to {val_scores[-1]}")
                #    torch.save(self.model.state_dict(), self.model_path)
#
        self.timing.update("train", self.timing.__getitem__("epoch"))
        self.timing.compute_per_epochs_timing(epoch+1, num_batches=1)

        return losses, val_scores, val_losses
    

    def test(
        self,
        x,
        y,
        metric = "accuracy",
        batch_size = 512
    ):
        self.model.eval()
        self.model.to(self.test_device)
        tic = time()
        n_batches = math.ceil(x.size(0) / batch_size)
        y_true = y if isinstance(y, torch.Tensor) else y.get_plain_text()
        y_pred = []
        for i in range(n_batches):
            x_batch = x[i*batch_size:(i+1)*batch_size].to(self.test_device)
            logits = self.model(x_batch)

            y_pred += [l.argmax().item() for l in logits]

        y_pred = torch.tensor(y_pred).to(self.device)
        if metric == "accuracy":
            test_score = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            test_score = f1_score(y_true, y_pred, average="micro")
        elif metric == "mcc":
            test_score = matthews_corrcoef(y_true, y_pred)
        else:
            raise ValueError(f"Invalid metric: {metric}")
        toc = time()
        print(f"Test {metric.capitalize()}: {test_score} Time: {toc-tic} s")
        self.model.to(self.device)
        
        return test_score


    def train_non_dp(
        self,
        x,
        y,
        x_val,
        y_val,
        validate = True,
        test = True,
        eval_one_batch = False,
        test_epochs = 5,
        batched = False
    ):
        losses = []
        test_scores = []

        if batched:
            for epoch in range(self.num_epochs):
                tic_epoch = time()
                self.model.train()
                for i in range(0, x.size(0), self.batch_size):
                    tic_batch = time()
                    x_batch = x[i:i+self.batch_size]
                    y_batch = y[i:i+self.batch_size]

                    tic = time()
                    logits = self.model(x_batch)
                    toc = time()
                    self.timing.add("forward", toc - tic)

                    tic = time()
                    loss = self.loss_fn(logits, y_batch)
                    toc = time()
                    self.timing.add("loss", toc - tic)

                    tic_grad = time()
                    tic = time()
                    loss.backward()
                    toc = time()
                    self.timing.add("backward", toc - tic)

                    tic = time()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    toc = time()
                    self.timing.add("param_update", toc - tic)
                    toc_grad = time()
                    self.timing.add("grad_compute", toc_grad - tic_grad)

                    print(f"\t\tbatch {i} loss {loss}")
                    toc_batch = time()
                    losses.append(loss)
                    self.timing.add("batch", toc_batch - tic_batch)
                    if eval_one_batch:
                        break

                toc_epoch = time()
                self.timing.add("epoch", toc_epoch - tic_epoch)
                print(f"Epoch {epoch} Loss {loss}")
                
                if validate:
                    val_score, val_loss = self.validate(x_val, y_val)
                    print(f"Validation {val_score} Loss {val_loss}")
        else:
            for epoch in range(self.num_epochs):
                tic_epoch = time()
                self.model.train()
                for i in range(0, x.size(0), self.batch_size):
                    x_batch = x[i:i+self.batch_size]
                    y_batch = y[i:i+self.batch_size]
                    for p in self.model.parameters():
                        p.accumulated_grads = 0

                    samples_loss = 0
                    batch_loss = 0

                    for j in range(x_batch.size(0)):
                        x_i = x_batch[j].unsqueeze(0)
                        y_i = y_batch[j].unsqueeze(0)

                        tic = time()
                        logits = self.model(x_i)
                        toc = time()
                        self.timing.add("forward", toc - tic)

                        tic = time()
                        loss_value = self.loss_fn(logits, y_i)
                        toc = time()
                        self.timing.add("loss", toc - tic)
   
                        with crypten.no_grad():
                            samples_loss += loss_value


                        tic = time()
                        loss_value.backward()
                        for p in self.model.parameters():
                            p.accumulated_grads += p.grad
                        toc = time()
                        self.timing.add("backward", toc - tic)


                    tic = time()
                    for p in self.model.parameters():
                        p.grad = p.accumulated_grads / x_batch.size(0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()            
                    toc = time()
                    self.timing.add("param_update", toc - tic)
                if validate:
                        val_score, val_loss = self.validate(x_val, y_val)
                        print(f"Validation {val_score} Loss {val_loss}")
        self.timing.update("train", self.timing.__getitem__("epoch"))
        self.timing.compute_per_epochs_timing(self.num_epochs, num_batches=1 if eval_one_batch else math.ceil(x.size(0) / self.batch_size))

        return losses



if __name__ == "__main__":
    crypten.init()
