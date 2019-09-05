import numpy as np
import random
import math
import time
import statistics
import matplotlib.pyplot as plt
import loss
import model
# import imagehash
# %matplotlib inline
# !pip install ImageHash
# random.seed(42)


# class ConfigModel:
#     def __init__(self):
#         self.config_pipe = 
        
#     def configure_model(self):
#         pass
    
#     def add_loss(self):
#         self.config_pipe
#         return self
    
#     def add_history(self):
#         return self
    
#     def add_stopping(self):
#         return self

#     def configure_bot(self):
#         return self
    
#     def create_population(self):
#         return self


# class AlphaLoss(loss.GeneticBaseLoss):
    # def __init__(self, alphabet, **kwargs):
    #     self.alphabet = alphabet
    #     super().__init__(**kwargs)

def compute(weights, target=None):
    # return sum([abs(self.alphabet.find(self.target[i]) - self.alphabet.find(c)) for i,c in enumerate(weights)]) / len(self.target)**2
    return np.sum(np.array([abs(target[i] - w) for i, w in np.ndenumerate(weights)])) / len(target)**2


alphabet = ' йцукенгшщзхъфывапролджэячсмитьбю0123456789!?+=-)(,.'
alphabet = ''.join(random.sample(alphabet, len(alphabet) ))
print(alphabet)

target = '''тест алгоритма ооочень длинным сообщением'''

bot_len = len(target)
nbots = 20000
nsurv = 5000
nnew = nbots - nsurv
mut = 0.2
nparents = min(500, nsurv)
epochs = 10

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
