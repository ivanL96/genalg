import numba
from numba import cuda, jit, njit

import random
import time
import numpy as np

# @jit(nopython=True)
# def monte_carlo_pi(nsamples):
#     acc = 0
#     for i in range(nsamples):
#         x = random.random()
#         y = random.random()
#         if (x ** 2 + y ** 2) < 1.0:
#             acc += 1
#     return 4.0 * acc / nsamples

# start = time.time()
# monte_carlo_pi(1000000)
# print(time.time() - start)

@jit(nopython=True, cache=True)
def create_next_population(n, vals, sorted_vals, population):
    out = []
    for i in range(n):
        out.append(population[vals.index(sorted_vals[i])])
    return out


def compute_loss(loss_function, p, bot_len, target=[]):
    return jit_compute_loss(njit(loss_function), p, bot_len, target)    


@jit(nopython=True, cache=True)
def jit_compute_loss(j_loss_function, p, bot_len, target=[]):
    # losses = [ j_loss_function(p[bot_len*i : (bot_len*i)+bot_len], target) for i,_ in enumerate(p) if bot_len*i < len(p)]
    losses = []
    for i,_ in enumerate(p):
        if bot_len*i < len(p):
            losses.append(j_loss_function(np.asarray(p[bot_len*i : (bot_len*i)+bot_len]), target))
    return losses


@jit(nopython=True, cache=True)
def test(l, h):
    print(l,h)
    sv = np.sum(np.random.randint(l, h, 1000000))
    # sv = sorted([random.randint(-100, 100000) for _ in range(1000000)])
    return sv

if __name__ == '__main__':
    start = time.time()
    test(np.random.randint(-100, 0), np.random.randint(1, 100000))
    print(time.time() - start)