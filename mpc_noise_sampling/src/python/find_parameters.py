
from utils.dp_utils import ddgauss_params, skellam_params, skellam_local_stddev, ddgauss_local_stddev
from time import time 
from fractions import Fraction
import utils.ibm_eps_rho_conv as cdp2adp


eps = 1
delta = 1e-6
dim = 1
steps = 1
l2_norm_clip = 0.1
bit = 16
num_clients = 1
subsamplig_parameter = 1/100

print(f"Field {2**bit}")

gamma, local_stddev = ddgauss_params(
                    q = subsamplig_parameter,
                    epsilon = eps,
                    l2_clip_norm = l2_norm_clip,
                    bits = bit,
                    num_clients = num_clients,
                    dim = dim,
                    delta = delta,
                    beta = 0,
                    steps = steps )


print(f"Google discrete Gaussian approx:\n \t gamma = {gamma} \n\t local_stddev = {local_stddev} \n\t scale = {local_stddev/gamma} \n\t scale^2 = {(local_stddev/gamma)**2}")


scale = 2**(bit)
gamma = 1/scale
local_stddev = ddgauss_local_stddev(
                    q = subsamplig_parameter,
                    gamma = gamma,
                    epsilon = eps,
                    l2_clip_norm = l2_norm_clip,
                    #bits = bit,
                    num_clients = num_clients,
                    dim = dim,
                    delta = delta,
                    beta = 0,
                    steps = steps )


print(f"Google discrete Gaussian approx with fixed scale {scale}:\n \t gamma = {gamma} \n\t local_stddev = {local_stddev} \n\t scale = {local_stddev/gamma} \n\t scale^2 = {(local_stddev/gamma)**2}, {(local_stddev/gamma)**2 < 2**bit}")


rho=cdp2adp.cdp_rho(eps,delta)
#compute noise variance parameter per query
# k=1
# rho_per_q = Fraction(rho)/k 
# sigma2=1/(2*rho_per_q)
sigma2=1/(2*rho)
print(f"IBM discrete Gaussian exact (1 query): \n\t scale = {sigma2}")



scale, mu = skellam_params(
                    q = subsamplig_parameter,
                    epsilon = eps,
                    l2_clip = l2_norm_clip,
                    bits = bit/2,
                    num_clients = num_clients,
                    dim = dim,
                    delta = delta,
                    beta = 0,
                    steps = steps )

print(f"Google Skellam:\n \t scale = {scale} \n\t mu = {mu} \n\t input mu = {mu*(scale**2)}")


scale = 2**(bit/2)
mu = skellam_local_stddev(
                    q = subsamplig_parameter,
                    epsilon = eps,
                    scale = scale,
                    l2_clip = l2_norm_clip,
                    #bits = bit,
                    num_clients = num_clients,
                    dim = dim,
                    delta = delta,
                    beta = 0,
                    steps = steps )

print(f"Google Skellam fixed scale :\n \t scale = {scale} \n\t mu = {mu} \n\t input mu = {mu*(scale**2)}, {mu*(scale**2)  < 2**bit}")


