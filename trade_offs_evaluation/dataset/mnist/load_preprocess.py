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


# Create data loaders for batching and shuffling
#batch_size = 64

#mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
#mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


def load_and_preprocess_mnist(
        scaling: bool = True,
        normalization: bool = False
):
    # Define the transformations: scaling and normalization
    #transform = transforms.Compose([
    #    transforms.ToTensor(),  # Scales pixel values to [0, 1]
    #    transforms.Normalize((0.1307,), (0.3081,))  # Normalizes with mean and std
    #])

    # Load the MNIST dataset with the defined transformations
    mnist_train = datasets.MNIST(root='dataset/mnist', train=True, download=True)#, transform=transform)
    mnist_test = datasets.MNIST(root='dataset/mnist', train=False, download=True)#, transform=transform)


    if scaling:
        if normalization:
            mnist_mean = torch.mean(mnist_train.data.float()) / 255
            mnist_std = torch.std(mnist_train.data.float()) / 255
        else:
            mnist_mean = 0
            mnist_std = 1

        # Transformation applied now since in MPC no DataLoader is used
        mnist_train.data = (mnist_train.data / 255 - mnist_mean) / mnist_std
        mnist_test.data = (mnist_test.data / 255 - mnist_mean) / mnist_std

    return mnist_train, mnist_test


def load_and_preprocess_mnist_federated(
    samples_per_user: int = 10
):
    
    mnist_train, mnist_test = load_and_preprocess_mnist()
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
