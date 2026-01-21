"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

import torch
from torchvision import datasets, transforms

from pfl.data import ArtificialFederatedDataset, get_data_sampler



def load_and_preprocess_fashion_mnist(
        scaling: bool = True,
        normalization: bool = False
):
    # Define the transformations: scaling and normalization


# Load the Fashion MNIST dataset with the same transformations
    fashion_mnist_train = datasets.FashionMNIST(root='dataset/fashion_mnist', train=True, download=True) #, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root='dataset/fashion_mnist', train=False, download=True) #, transform=transform)

    if scaling:
        if normalization:
            fashion_mnist_mean = torch.mean(fashion_mnist_train.data.float()) / 255
            fashion_mnist_std = torch.std(fashion_mnist_train.data.float()) / 255
        else:
            fashion_mnist_mean = 0
            fashion_mnist_std = 1

        # Transformation applied now since in MPC no DataLoader is used
        fashion_mnist_train.data = (fashion_mnist_train.data / 255 - fashion_mnist_mean) / fashion_mnist_std
        fashion_mnist_test.data = (fashion_mnist_test.data / 255 - fashion_mnist_mean) / fashion_mnist_std

    return fashion_mnist_train, fashion_mnist_test


def load_and_preprocess_fashion_mnist_federated(
    samples_per_user: int = 10,
    scaling: bool = True,
    normalization: bool = False
):
    
    mnist_train, mnist_test = load_and_preprocess_fashion_mnist(
        scaling=scaling,
        normalization=normalization
    )
    
    data_sampler = get_data_sampler(sample_type="random", max_bound=mnist_train.data.shape[0])

    data_sampler_val = get_data_sampler(sample_type="random", max_bound=mnist_test.data.shape[0])
    
    sample_dataset_len = lambda: samples_per_user
    mnist_federated_train = ArtificialFederatedDataset.from_slices(
        data = [mnist_train.data, mnist_train.targets.squeeze()],
        data_sampler = data_sampler,
        sample_dataset_len = sample_dataset_len
    )

    mnist_federated_val = ArtificialFederatedDataset.from_slices(
        data = [mnist_test.data, mnist_test.targets.squeeze()],
        data_sampler = data_sampler_val,
        sample_dataset_len = sample_dataset_len
    )

    return mnist_federated_train, mnist_federated_val, mnist_test
