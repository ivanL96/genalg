import numpy as np
from random import random, sample
import math
import time
import statistics
import matplotlib.pyplot as plt

from ga.model import *
from ga.preprocessing import *

# import imagehash


def compute(weights, target=None):
    # return sum([abs(self.alphabet.find(self.target[i]) - self.alphabet.find(c)) for i,c in enumerate(weights)]) / len(self.target)**2
    # return sum([abs(target[i] - weights[i]) for i in range(len(weights)) ])**2
    return abs(max(weights) - target)


# alphabet = ' йцукенгшщзхъфывапролджэячсмитьбю0123456789!?+=-)(,.'
# alphabet = randomize_string(alphabet)
# target = '''тест алгоритма'''
alphabet=None
target = 1000

bot_len = 10#len(target)
nbots = 10000
nsurv = 2000
nnew = nbots - nsurv
mut = 0.2
nparents = min(100, nsurv)
epochs = 1000

if __name__ == '__main__':
    genmod = GeneticModel(nbots, bot_len, nsurv, nnew, 
                    nparents, mut, alphabet=alphabet, 
                    target=target)
    genmod.configure_bot(weight_type=float, weight_range=(0,2000))
    genmod.add_loss(compute)
    genmod.add_stopping('best', 0)
    genmod.run(epochs=epochs, init_multiplier=1, verbose=1)
