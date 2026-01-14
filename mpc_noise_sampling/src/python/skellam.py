from time import time
import numpy as np

"""
Transformed rejection due to Hormann.
Given a CDF F(x), and G(x), a dominating distribution chosen such
that it is close to the inverse CDF F^-1(x), compute the following
steps:
    1) Generate U and V, two independent random variates. Set U = U - 0.5
        (this step isn't strictly necessary, but is done to make some
        calculations symmetric and convenient. Henceforth, G is defined on
        [-0.5, 0.5]).
    2) If V <= alpha * F'(G(U)) * G'(U), return floor(G(U)), else return
        to step 1. alpha is the acceptance probability of the rejection
        algorithm.
For more details on transformed rejection, see:
http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.

The dominating distribution in this case:
G(u) = (2 * a / (2 - |u|) + b) * u + c
"""

"""
def poisson_hormann(lam):
    # Step 1
    b = math.pi / math.sqrt(3.0 * lam)
    a = b * lam
    c = 0.767 - 3.36 / lam
    k = math.log(c) - lam - math.log(b)
    m = 0
    n = 0

    while True:
        # Step 2
        u = random.random()
        x = (a - math.log(1.0 - u)) / b
        n = math.floor(x + 0.5)
        if n < 0:
            continue
        v = random.random()
        # Step 3
        y = a - b * x
        lhs = y + math.log(v / (1.0 + ((x - m) / (n - m)) ** 2))
        rhs = k + n * math.log(lam) - math.lgamma(n + 1)
        if lhs <= rhs:
            return n

def poisson_hormann(lam):
    b = math.pi / math.sqrt(3.0 * lam)
    a = 0.767 - 3.36 / lam
    k = math.log(b) - lam - math.log(a)

    while True:
        u = random.random()
        x = (a - math.log(1.0 - u)) / b
        n = math.floor(x + 0.5)
        if n < 0:
            continue
        v = random.random()
        y = ((x - n)**2) / 2
        lhs = k + n * math.log(lam) - math.lgamma(n + 1)
        rhs = math.log(v) + y
        if lhs >= rhs:
            return n



cdef double draw_one(double alpha, double beta, double k, double loglam):
    cdef int cost = 0
    cdef double x, y, lhs, rhs
    cdef float u, v
    cdef int n

    while True:
        u = random_sample()
        cost += 1
        x = (alpha - np.log(1-u)/u)/beta
        n = int(np.floor(x + 0.5))
        if n < 0:
            continue
        v = random_sample()
        cost += 1
        y = alpha - beta*x
        lhs = y + np.log(v/(1 + np.exp(y))**2)
        rhs = k + n*loglam - gammaln(n+1)
        if lhs <= rhs:
            return n
"""


def poisson_uni(lambda_):
    k = -1
    p = 1.
    lambda_exp = np.exp(-lambda_)
    while p > lambda_exp:
        k += 1
        p *= np.random.rand()
    return k


lam = 1946294362
n_iterations = lam + 10
tic = time ()
sk = poisson_uni(lam/2) - poisson_uni(lam/2)
toc = time()
print(f"Avg time = {(toc-tic)}")
print(sk)