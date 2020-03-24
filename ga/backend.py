import random
import time
from os.path import samefile

import numba
import numpy as np
from numba import cuda, jit, njit
from numba.types import List, byte, float32, float64, int32, int64, pyfunc_type

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
    # @njit('f8(f8)')
    # def lol(x):
    #     return np.sin(x)    
    j_loss_function = njit(f8(f8[:], f8[:]), cache=True)(loss_function)
    return jit_compute_loss(j_loss_function, np.asarray(p), bot_len, np.asarray(target))    

@njit
def jit_compute_loss(j_loss_function, p, bot_len, target):
    losses = [ j_loss_function(p[bot_len*i : (bot_len*i)+bot_len], target) for i,_ in enumerate(p) if bot_len*i < len(p)]
    # losses = []
    # for i,_ in enumerate(p):
    #     if bot_len*i < len(p):
    #         losses.append(j_loss_function(np.asarray(p[bot_len*i : (bot_len*i)+bot_len]), target))
    return 0.0


def create_new_bot(nnew, nsurv, nparents, bot_len, next_population, mut, weight_sample_enum):

    # sample_random_types = [random.randint, np.random.uniform, random.sample]
    rand_type = weight_sample_enum[0]
    rand_range = np.array(weight_sample_enum[1])
    rand_sample_n = weight_sample_enum[2]

    next_population = np.asarray(next_population, dtype=np.int16) # type hardcoded!!!
    return jit_create_new_bot(nnew, nsurv, nparents, bot_len, mut, next_population, 
            rand_type, rand_range, rand_sample_n)

# @njit(float32[:](int32, int32, int32, int32, float32[:], float32, byte, float32[:], byte), cache=True)
@njit(cache=True)
def jit_create_new_bot(nnew, nsurv, nparents, bot_len, mut, next_population, rand_type, rand_range, rand_sample_n): 
    bot = []
    # bot = []
    parents = [ next_population[random.randint(0,nsurv-1)] for parent in range(nparents) ]
    for n in range(bot_len):
        dominant = random.randint(0, nparents-1)
        if random.uniform(0, 1) < mut:
            if rand_type == 0:
                weight = random.randint(rand_range[0], rand_range[1])
            if rand_type == 1:
                weight = np.random.uniform(rand_range[0], rand_range[1])
            if rand_type == 2:
                weight = np.random.choice(rand_range, rand_sample_n)[0]
        else:
            weight = parents[dominant][n]
        # np.append(bot, weight)
        bot.append(weight)
    return bot




def run_epoch(nbots, nsurv, nnew, mut, weight_sample, bot_len):
    pass














# @jit(nopython=True, cache=True)
# def test(l, h):
#     print(l,h)
#     sv = np.sum(np.random.randint(l, h, 1000000))
#     # sv = sorted([random.randint(-100, 100000) for _ in range(1000000)])
#     return sv

# if __name__ == '__main__':
#     start = time.time()
#     test(np.random.randint(-100, 0), np.random.randint(1, 100000))
#     print(time.time() - start)
