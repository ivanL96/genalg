from numba import cuda, jit
import random
import time

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
