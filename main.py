import numpy as np
from random import random, sample
import math
import time
import statistics
import matplotlib.pyplot as plt

from ga.model import *
from ga.preprocessing import *

# import imagehash
# random.seed(42)


def compute(weights, target=None):
    # return sum([abs(self.alphabet.find(self.target[i]) - self.alphabet.find(c)) for i,c in enumerate(weights)]) / len(self.target)**2
    return sum([abs(target[i] - weights[i]) for i in range(len(weights)) ])**2


alphabet = ' йцукенгшщзхъфывапролджэячсмитьбю0123456789!?+=-)(,.'
alphabet = ''.join(sample(alphabet, len(alphabet) ))  # randomize
print(alphabet)

target = '''тест алгоритма'''

bot_len = len(target)
nbots = 5000
nsurv = 1000
nnew = nbots - nsurv
mut = 0.2
nparents = min(100, nsurv)
epochs = 1000

gen = GeneticModel(nbots, bot_len, nsurv, nnew, 
                   nparents, mut, alphabet=alphabet, 
                   target=target)
gen.configure_bot(weight_type=str)
gen.add_loss(compute)
gen.add_stopping('best', 0)

gen.run(epochs=epochs, n=1, verbose=1)
