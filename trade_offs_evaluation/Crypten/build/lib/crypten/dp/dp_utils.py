from time import time
import torch

from .skellam_noise import skellam_params, skellam_local_stddev
from .gaussian_noise import get_gaussian_stddev
from ..common.functions.sampling import randn




def l2_norm_clip(x, clip_threshold):
    """ This should me an MPC function """
    #TODO: test how it works with crypten tensors
    # crypten.print(f"Grad shape: {d_p.shape}")
    #dim = 1 % len(x.shape)
    #if not dim:
    #    l2_norm = x.abs()
    #else:
    l2_norm = x.norm(p=2, keepdim=True)
    inv_clip= 1/clip_threshold
    clip = l2_norm.gt(clip_threshold)
    x = (x / (l2_norm*inv_clip) - x) * clip + x


def l2_norm_clip_optim(x, clip_threshold):
    """ This should me an MPC function """
    squared_x_sum = x.square().sum()
    inv_l2_norm = squared_x_sum.inv_sqrt()
    tmp = inv_l2_norm * clip_threshold
    clip = (tmp).lt(1)
    x = (x * tmp - x) * clip + x


def compute_stddev_from_epsilon(
        #noise_type, 
        noise_mechanism, 
        eps, 
        delta, 
        clipping_threshold, 
        num_parties = 1, 
        steps = 1,
        sampling_rate = 1.0,
        **kwargs
        ):
    """ This should be a local function """
    if noise_mechanism == "gaussian":
        return get_gaussian_stddev(
            eps = eps,
            delta = delta,
            sampling_rate = sampling_rate,
            steps = steps,
            clipping_threshold = clipping_threshold
            )/num_parties
    elif noise_mechanism == "skellam":
        scale = kwargs.get("scale", None)
        beta = kwargs.get("beta", 1.0)
        dim = kwargs.get("dim", 1)
        
        if scale is None:
            k = kwargs.get("k", 3)
            rho = kwargs.get("rho", 1)
            sqrtn_norm_growth = kwargs.get("sqrtn_norm_growth", False)
            scale, stddev = skellam_local_stddev(
                epsilon = eps,
                l2_clip=clipping_threshold,
                num_clients=num_parties,
                beta=beta,
                dim=dim,
                q=sampling_rate,
                steps=steps,
                delta=delta,
                k = k,
                rho = rho,
                sqrtn_norm_growth = sqrtn_norm_growth
            )
            return stddev*scale
        
        return scale * skellam_local_stddev(
            epsilon = eps,
            scale = scale,
            l2_clip=clipping_threshold,
            num_clients=num_parties,
            beta=beta,
            dim=dim,
            q=sampling_rate,
            steps=epochs,
            delta=delta,
        )
    else:
        raise ValueError(f"Invalid noise mechanism: {noise_mechanism}")



def sample_skellam_noise_local(shape, local_stddev):
    """ This should be a local function """
    """
        Function from https://github.com/google-research/federated/blob/master/distributed_dp/distributed_skellam_query.py
    """
    # the precision should be 16 bits by default in Crypten/configs/default.yaml
    # to define custom configuration create a new yaml file and put into the data_files
    shape = (shape, 2) # the last dimension is for the two poisson distributions to get the skellam distribution
    local_stddev = torch.tensor(local_stddev, dtype=torch.float64)
    poisson_lambda = 0.5 * local_stddev * local_stddev
    seed = int(time() * 10**6)
    torch.manual_seed(seed)
    poissons = torch.poisson(torch.tensor([poisson_lambda, poisson_lambda]).repeat(*shape[:-1], 1))
    return (poissons[..., 0] - poissons[..., 1]).to(torch.float64)


def sample_gaussian_noise_local(shape, local_stddev):
    """ This should be a local function """
    local_stddev = torch.tensor(local_stddev, dtype=torch.float64)
    seed = int(time() * 10**6)
    torch.manual_seed(seed)
    return torch.normal(mean=0, std=local_stddev, size=shape).to(torch.float64)
    

def sample_gaussian_noise_global(shape, global_stddev):
    """ This should be an MPC function 
        We use the Box-Muller transform to sample 
        standard Gaussian noise and then we rescale it
        using the global standard deviation
        This function can return two samples at a time
        """
    # Box-Muller transform with torch functions
    #u1 = torch.rand(shape)
    #u2 = torch.rand(shape)
    #z1 = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * 3.141592653589793 * u2)
    #z2 = torch.sqrt(-2 * torch.log(u1)) * torch.sin(2 * 3.141592653589793 * u2)
    #return z1 * global_stddev, z2 * global_stddev

    # TODO: Currently returns only one sample
    return randn(shape)[0] * global_stddev


def sample_skellam_noise_global(shape, global_stddev):
    """ This should be a global function """
    #TODO: Implement the global noise sampling
    raise NotImplementedError("Global noise sampling is not implemented yet")


def sample_local_noise(shape, local_std, noise_mechanisms="gaussian"):
    switch = {
        "gaussian": sample_gaussian_noise_local,
        "skellam": sample_skellam_noise_local
    }

    try:
        return switch[noise_mechanisms](shape, local_std)
    except KeyError:
        raise ValueError(f"Invalid noise type: {noise_mechanisms}")
    

def sample_global_noise(shape, global_std, noise_mechanism="gaussian"):
    """ This should be a global function """
    switch = {
        "gaussian": sample_gaussian_noise_global,
        "skellam": sample_skellam_noise_global
    }
    
    if noise_mechanism not in switch:
        raise ValueError(f"Invalid noise mechanism: {noise_mechanism}")
    # TODO: Implement the global noise sampling
    return switch[noise_mechanism](shape, global_std)
    """ How to sample uniform random value for the Box-Muller transform?"""