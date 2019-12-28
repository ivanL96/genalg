import numpy as np
import random
import math
import time
import statistics
import matplotlib.pyplot as plt
import loss
import model
# import imagehash
# random.seed(42)


# class AlphaLoss(loss.GeneticBaseLoss):
    # def __init__(self, alphabet, **kwargs):
    #     self.alphabet = alphabet
    #     super().__init__(**kwargs)

def compute(weights, target=None):
    # return sum([abs(self.alphabet.find(self.target[i]) - self.alphabet.find(c)) for i,c in enumerate(weights)]) / len(self.target)**2
    return sum([abs(target[i] - weights[i]) for i in range(len(weights)) ])**2


alphabet = ' йцукенгшщзхъфывапролджэячсмитьбю0123456789!?+=-)(,.'
alphabet = ''.join(random.sample(alphabet, len(alphabet) ))
print(alphabet)

target = '''тест алгоритма'''
import preprocessing
ohe = preprocessing.to_ohe(target)
print(ohe)
print(preprocessing.normalize(ohe))

bot_len = len(target)
nbots = 5000
nsurv = 1000
nnew = nbots - nsurv
mut = 0.2
nparents = min(100, nsurv)
epochs = 1000

gen = model.GeneticModel(nbots, bot_len, nsurv, nnew, 
                         nparents, mut, alphabet=alphabet, 
                         target=target)
gen.configure_bot(weight_type=str)
gen.add_loss(compute)
# gen.mutate_nsurv()
# gen.mutate_nnew()
# gen.mutate_nparents()
gen.add_stopping('best', 0)

gen.run(epochs=epochs, n=1, verbose=1)
