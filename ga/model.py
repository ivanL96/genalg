    
from time import perf_counter
from random import seed, random, uniform, randint, sample
import numpy as np

seed(0)

from functools import lru_cache

from . import backend


def print_time(*args):
    print(*args)
    return None


def _randint(a,b):
    return int(max(a, random()*b))


class ModelConfiguration:
    def __init__(self, nbots, bot_len, nsurv, nnew, nparents, mut=0.1, reserved_weights=0, alphabet=None, target=None):
        self.population = []
        self.next_population = []

        # bot params
        self.weight_sample = None
        self.weight_sample_enum = None
        self.bot_len = bot_len
        self.reserved_weights = reserved_weights
        
        # population params
        self.nbots = nbots
        self.nsurv = nsurv
        self.nnew = nnew
        self.nparents = nparents
        self.mut = mut
        
        # history
        self.history = {
            'best': [],
            'mean': [],
            'pstdev': [],
            'pvariance': [],
            
            'stdev': [],
            'variance': [],
            
            'best_worst': [],            
        }

        self.target = target
        self.alphabet = None
        # alphabet is a sequence of symbols that bots takes for setting its weights. 
        # If None - consider as sequence of real numbers
        if alphabet:
            self.raw_alphabet = alphabet
            self.alphabet = [i for i, ch in enumerate(self.raw_alphabet)]
            self.alphabet_len = len(self.alphabet)
            if target:
                self.raw_target = target
                try:
                    self.target = [self.raw_alphabet.index(ch) for ch in target]
                except ValueError:
                    raise ValueError('The target has symbols not present in alphabet')

        self.stoppings = dict()
        self.loss_function = None
        print('Configuration: bot_len-{}, nbots-{}, nsurv-{}, nnew-{}, nparents-{}, mut-{}'
              .format(bot_len, nbots, nsurv, nnew, nparents, mut))

    # configuration methods=====================================================
    def add_loss(self, loss_function):
        self.loss_function = loss_function
        if not callable(self.loss_function):
            raise ValueError('loss_function parameter must be callable.')
        return self
    
    def add_stopping(self, history_name, target_value):
        if history_name in self.history:
            self.stoppings[history_name] = target_value
        else:
            avail_h = ', '.join(self.history.keys())
            msg = 'Cannot find history with name {}. Available histories are: {}'
            raise ValueError(msg.format(history_name, avail_h))
        return self
            
    def configure_bot(self, weight_type=int, weight_range=(0,1)):
        """weight_type can be str, int, float"""
        if weight_type == int:
            self.weight_sample = (_randint, weight_range)
            self.weight_sample_enum = (0, weight_range, 0)
        elif weight_type == float:
            self.weight_sample = (np.random.uniform, weight_range)
            self.weight_sample_enum = (1, weight_range, 0)
        elif weight_type == str:
            if self.alphabet is not None:
                self.weight_sample = (sample, self.alphabet, 1)
                self.weight_sample_enum = (2, self.alphabet, 1)        
            else:
                raise ValueError('Weight type specified as string, but the alphabet not set.')
        else:
            raise TypeError('Unknown weight type. Use str, int or float.')
        return self


class GeneticModel(ModelConfiguration):
    def __init__(self, nbots, bot_len, nsurv, nnew, nparents, mut=0.1, reserved_weights=0, alphabet=None, target=None):
        super().__init__(nbots, bot_len, nsurv, nnew, nparents, mut, reserved_weights, alphabet, target)
        
    
    # auxiliary methods========================================================= 
    def __get_bot(self, bot):
        if self.reserved_weights:
            return bot[:-self.reserved_weights]
        return bot
    
    def __get_sample(self):
        rand_func = self.weight_sample[0]
        args = self.weight_sample[1:]
        if args[0] is None:
            raise ValueError('The weight_range parameter at configure_bot() has invalid value.')
        if len(args) == 1:
            args = args[0]
        
        # optimization of rands
        if self.weight_sample_enum[0] == 2: # i.e. random.sample
            # sample = self.alphabet[int(random() * self.alphabet_len)]
            sample = int(random() * self.alphabet_len)
        else:
            sample = rand_func(*args)

        sample = sample[0] if type(sample) == list else sample
        return sample
        
    @lru_cache(maxsize=4096)
    def __get_loss(self, bot):
        return self.loss_function(bot, self.target)

    def _check_stoppings(self):
        result=[]
        for stop_name, stop_val in self.stoppings.items():
            if len(self.history[stop_name]):
                result.append(stop_val == self.history[stop_name][-1])
            else:
                result.append(False)
        return result
    

    # main methods==============================================================
    def random_bot(self):
        return [self.__get_sample() for el in range(self.bot_len)]

    def __new_bot(self):
        # creates a new bot for a new generation
        # mutations = 0
        bot = []
        parents = [ self.next_population[int(random()*(self.nsurv-1))] for parent in range(self.nparents) ]
        _nprts = self.nparents-1
        _mut = self.mut
        for n in range(self.bot_len):
            dominant = int(random() * _nprts)

            if uniform(0, 1) < _mut:  # apply mutation to current weight
                weight = self.__get_sample()
            else:
                weight = self.__get_bot(parents[dominant])[n]
            bot.append(weight)

        # self.history['pop_mutations'].append(mutations)
        return bot

    def __bot2text(self, bot):
        if self.alphabet:
            return ''.join(self.raw_alphabet[w] for w in bot)
        return '_'.join(str(w) for w in bot)

    def run(self, epochs, init_multiplier=1, verbose=1):
        times=[]
        if self.weight_sample == None:
            raise ValueError('Bot weights must be configured before creating the population. Use gen.configure_bot().')
        
        # creating 1D population
        self.population = [self.random_bot() for _ in range(self.nbots * init_multiplier)]
        _nsrv = self.nsurv
        _nnw = self.nnew
        for it in range(epochs):
            start = perf_counter()
            
            vals = [(bot, self.loss_function( bot, self.target )) for bot in self.population]

            sorted_vals = sorted(vals, key=lambda x: x[1])
            best_loss_res = sorted_vals[0][1]
            # visualization
            self.history['best'].append(best_loss_res)
            # self.history['mean'].append(np.mean(vals))
            # self.history['best_worst'].append(abs(sorted_vals[0]-sorted_vals[-1]))
            # self.history['pstdev'].append(statistics.pstdev(sorted_vals))
            # self.history['pvariance'].append(statistics.pvariance(sorted_vals))
            # self.history['stdev'].append(statistics.stdev(sorted_vals))
            # self.history['variance'].append(statistics.variance(sorted_vals))
            # self.history['gen_mut_history'].append(np.mean(gen_mutation))

            # get surv bots
            self.next_population = [el[0] for el in sorted_vals[:_nsrv]]

            worst_bot = self.next_population[-1] # for printing

            # !!!!!!!!!!!!EXPENSIVE!!!!!!!!!!!!
            app_time = perf_counter()
            self.next_population += [self.__new_bot() for i in range(_nnw)]
            self.population = self.next_population


            et = round(perf_counter()-start, 3)
            times.append(et)
            if verbose == 1:
                print(it, 'loss: ', best_loss_res, 'bot: ', 
                    self.__bot2text(self.next_population[0]),
                    # self.__bot2text(worst_bot),
                    et)
            
            if any(self._check_stoppings()):
                break
            
        print('mean generation time: {} sec'.format(np.mean(times)))

