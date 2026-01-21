#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator
import torch
import time

from .optimizer import Optimizer
import crypten.communicator as comm
import crypten.dp as dp 

class DPSGD(Optimizer):
    epsilon = None
    delta = None

    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        grad_threshold (float, optional): imposes a threshold on the magnitude of gradient values.
            Gradient values with magnitude above the threshold will be replaced with 0.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        lr,
        noise_stddev,
        l2_clipping_threshold=1.0,
        noise_mechanism="gaussian",
        noise_type="local",
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        grad_threshold=None,
        device="cpu"
        #epsilon = 10.0,  # Privacy parameter
        #delta = 1e-5  # Privacy parameter (small constant)

    ):
        if not isinstance(lr, (int, float)) or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not isinstance(momentum, (int, float)) or momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not isinstance(dampening, (int, float)):
            raise ValueError("Invalid dampening value {}".format(dampening))
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        
        if not isinstance(noise_stddev, (int, float)) or noise_stddev < 0.0:
            raise ValueError("Invalid noidr std dev value: {}".format(noise_stddev))
        if not isinstance(l2_clipping_threshold, (int, float)) or l2_clipping_threshold < 0.0:
            raise ValueError("Invalid l2 clipping threshold value: {}".format(l2_clipping_threshold))

        self.noise_stddev = noise_stddev
        self.l2_clipping_threshold = l2_clipping_threshold
        self.noise_mechanism = noise_mechanism
        self.noise_type = noise_type

        self.noise_time = 0.0
        self.noise_communication = {
            "rounds": 0,
            "bytes": 0,
            "time": 0
        }

        self.param_update_time = 0.0
        self.param_update_communication = {
            "rounds": 0,
            "bytes": 0,
            "time": 0
        }

        self.device = device


        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # Compute thresholding based on square value since abs is more expensive
        self.square_threshold = grad_threshold
        if self.square_threshold is not None:
            self.square_threshold *= self.square_threshold

        super(DPSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DPSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def reset_noise_time(self):
        self.noise_time = 0.0

    def get_noise_time(self):
        return self.noise_time
    
    def reset_noise_communication(self):
        self.noise_communication = {
            "rounds": 0,
            "bytes": 0,
            "time": 0
        }

    def get_noise_communication(self):
        return self.noise_communication
    
    def add_noise_communication(self, noise_communication):
        self.noise_communication["rounds"] += noise_communication["rounds"]
        self.noise_communication["bytes"] += noise_communication["bytes"]
        self.noise_communication["time"] += noise_communication["time"]

    
    def reset_param_update_time(self):
        self.param_update_time = 0.0

    def get_param_update_time(self):
        return self.param_update_time

    def reset_param_update_communication(self):
        self.param_update_communication = {
            "rounds": 0,
            "bytes": 0,
            "time": 0
        }

    def get_param_update_communication(self):
        return self.param_update_communication
    
    def add_param_update_communication(self, param_update_communication):
        self.param_update_communication["rounds"] += param_update_communication["rounds"]
        self.param_update_communication["bytes"] += param_update_communication["bytes"]
        self.param_update_communication["time"] += param_update_communication["time"]

    def step(self, batch_size ,closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with crypten.no_grad():
            loss = None
            if closure is not None:
                with crypten.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    # Threshold gradients to prevent gradient explosion
                    if self.square_threshold is not None:
                        d_p = p.grad.mul(p.grad.square().lt(self.square_threshold))
                    else:
                        d_p = p.grad

                    if weight_decay != 0:
                        d_p = d_p.add(p.mul(weight_decay))
                    if momentum != 0:
                        param_state = self.state[id(p)]
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = d_p.clone().detach()
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(d_p.mul(1 - dampening))
                        if nesterov:
                            d_p = d_p.add(buf.mul(momentum))
                        else:
                            d_p = buf
                    
                    # Here apply DPSGD collaboratevely clipping and adding the right noise amount
                    #crypten.print(f"\t grad value optimizer: {d_p.get_plain_text()}")
                    if self.noise_type == "global":
                        crypten.communicator.get().reset_communication_stats()
                        tic = time.time()
                        dp_noise = dp.dp_utils.sample_global_noise(
                            shape = p.shape,
                            global_std = self.noise_stddev,
                            noise_mechanism=self.noise_mechanism
                        )
                        toc = time.time()
                        self.add_noise_communication(crypten.communicator.get().get_communication_stats())
                        self.noise_time += toc - tic
                    else:
                        rank = comm.get().get_rank()
                        crypten.communicator.get().reset_communication_stats()
                        tic = time.time()
                        local_noise = dp.dp_utils.sample_local_noise(
                            shape = p.shape, 
                            local_std=self.noise_stddev, 
                            noise_mechanisms=self.noise_mechanism
                        )
                        toc = time.time()
                        self.add_noise_communication(crypten.communicator.get().get_communication_stats())
                        self.noise_time += toc - tic
                        dp_noise = crypten.cryptensor(local_noise, src = rank)
                    
                    #rank = comm.get().get_rank()
                    #dp.dp_utils.l2_norm_clip(d_p, self.l2_clipping_threshold)
                    #local_skellam = dp.dp_utils.sample_skellam_noise(d_p,self.noise_local_stddev,clip_threshold=self.l2_clipping_threshold)
    
                    #skellam_noise_enc = crypten.cryptensor(local_skellam, src = rank)

                    #batch_inv = 1 / batch_size
                    crypten.communicator.get().reset_communication_stats()
                    tic = time.time()
                    d_p = (d_p + dp_noise.to(self.device)) / batch_size

                    p.sub_(d_p.mul(group["lr"]))
                    toc = time.time()
                    self.param_update_time += toc - tic
                    self.add_param_update_communication(crypten.communicator.get().get_communication_stats())

            return loss
