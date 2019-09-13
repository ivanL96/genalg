    
import time
import random
import numpy as np
from loss import GeneticBaseLoss
import backend


class Mutation:
    def __init__(self):
        self.muts = {
            'mut_nsurv': False,
            'mut_nnew': False,
            'mut_nparents': False
        }
        self.mut_nsurv = False
        self.mut_nnew = False
        self.mut_nparents = False

    def mutate_nsurv(self):
        self.muts['mut_nsurv'] = True
        return self

    def mutate_nnew(self):
        self.muts['mut_nnew'] = True
        return self

    def mutate_nparents(self):
        self.muts['mut_nparents'] = True
        return self

    def apply_mutation(self, mut):
        limit = int(mut * 10)
        nsurv = 0
        nnew = 0
        nparents = 0
        if self.muts['mut_nsurv']:
            nsurv += random.randint(-limit, limit)
        if self.muts['mut_nnew']:
            nnew += random.randint(-limit, limit)
        if self.muts['mut_nparents']:
            nparents += random.randint(-limit, limit)
        return nsurv, nnew, nparents


class GeneticModel:
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
            
#             'gen_mut_history': []
        }

        # alphabet is a sequence of symbols that bots takes for setting its weights. 
        # If None - consider as sequence of real numbers
        if alphabet:
            self.raw_alphabet = alphabet
            self.alphabet = [i for i, ch in enumerate(self.raw_alphabet)]
            if target:
                self.raw_target = target
                self.target = [self.raw_alphabet.index(ch) for ch in target]


        self.stoppings = dict()
        self.loss_function = None
        print('Configuration: bot_len-{}, nbots-{}, nsurv-{}, nnew-{}, nparents-{}, mut-{}'
              .format(bot_len, nbots, nsurv, nnew, nparents, mut))
        super().__init__()
        
        
    # configuration methods=====================================================
    def add_loss(self, loss_function):
        self.loss_function = loss_function#(target=self.target)
        # if not isinstance(self.loss_function, GeneticBaseLoss):
        #     raise ValueError('Loss class must be set. Use gen.add_loss().')
        return self
    
    # def add_history(self, history_obj):
    #     if isinstance(history_obj, History):
    #         self.history.update(history_obj.get_history())
    #     else:
    #         raise TypeError('Cannot use {} type as history object.'.format(type(history_obj)))
    #     return self
    
    def add_stopping(self, history_name, target_value):
        self.stoppings[history_name] = target_value
        return self
            
    def configure_bot(self, weight_type=int, weight_range=None):
        """weight_type can be str, int, float"""
        if weight_type == int:
            self.weight_sample = [random.randint, weight_range]
            self.weight_sample_enum = [0, weight_range, 0]
        elif weight_type == float:
            self.weight_sample = [np.random.uniform, weight_range]
            self.weight_sample_enum = [1, weight_range, 0]
        elif weight_type == str:
            if self.alphabet is not None:
                self.weight_sample = [random.sample, self.alphabet, 1]
                self.weight_sample_enum = [2, self.alphabet, 1]        
            else:
                raise ValueError('Weight type specified as string, but the alphabet not set.')
        else:
            raise TypeError('Unknown weight type. Use str, int or float.')
        return self
    
    
    # auxiliary methods========================================================= 
    def __get_bot(self, bot, all_weights=False, reserved=0):
        if self.reserved_weights:
            if reserved != 0:
                return bot[reserved]
            if not all_weights:
                return bot[:-self.reserved_weights]
        return bot
    
    def __get_sample(self):
        rand_func = self.weight_sample[0]
        args = self.weight_sample[1:]
        if args[0] is None:
            raise ValueError('The weight_range parameter at configure_bot() has invalid value.')
        if len(args) == 1:
            args = args[0]
        sample = rand_func(*args)
        sample = sample[0] if type(sample) == list else sample
        return sample
        
    def __get_loss(self, bot):
        return self.loss_function(bot, self.target)#.compute(bot)

    def _check_stoppings(self):
        return (stop_val == self.history[stop_name][-1] if len(self.history[stop_name]) else False for stop_name, stop_val in self.stoppings.items())
    
    
    # main methods==============================================================
    def random_bot(self):
        return [self.__get_sample() for el in range(self.bot_len)]# + [max(0.0001, random.uniform(-0.5, 0.5))]

    def __new_bot(self):
        # creates a new bot for a new generation
        # mutations = 0
        bot = []
        # parent_mut = []
        parents = [ self.next_population[random.randint(0,self.nsurv-1)] for parent in range(self.nparents) ]
        for n in range(self.bot_len):#len(main_parent)):
            dominant = random.randint(0, self.nparents-1)

            if random.uniform(0, 1) < self.mut:#self.__get_bot(parents[dominant], reserved=-1):
                weight = self.__get_sample()
                # mutations += 1
            else:
                weight = self.__get_bot(parents[dominant])[n]
                # parent_mut.append(self.__get_bot(parents[dominant], reserved=-1))
            bot.append(weight)

        # bot.append(random.sample(parent_mut, 1)[0])
        # self.history['pop_mutations'].append(mutations)
        return bot

    def __bot2text(self, bot):
        if self.alphabet:
            return ''.join(self.raw_alphabet[w] for w in bot)
        return ''.join(str(w) for w in bot)

    def run(self, epochs, n=1, verbose=1):
        times=[]
        if self.weight_sample == None:
            raise ValueError('Bot weights must be configured before creating the population. Use gen.configure_bot().')
        
        # creating 1D population
        # [self.population.extend(self.random_bot()) for _ in range(self.nbots * n)]
        self.population = [self.random_bot() for _ in range(self.nbots * n)]

        for it in range(epochs):
            start = time.time()
            # s, n, p = self.apply_mutation(self.mut)
            # self.nsurv += s 
            # self.nnew += n
            # self.nparents += p
            
            # gen_mutation = []
            loss_time = time.time()
            vals = [(bot, self.loss_function( bot, self.target )) for bot in self.population]
            # print('loss___', time.time()-loss_time)
            # vals = backend.compute_loss(self.loss_function, self.population, self.bot_len, self.target)
            # gen_mutation.append(self.__get_bot(bot, reserved=-1))

            # the lower score is the best
            sloss_time = time.time()
            sorted_vals = sorted(vals, key=lambda x: x[1])
            # print('sloss__', time.time()-sloss_time)

            # visualization
            self.history['best'].append(sorted_vals[0][1])
            # self.history['mean'].append(np.mean(vals))
            # self.history['best_worst'].append(abs(sorted_vals[0]-sorted_vals[-1]))
            
            # self.history['pstdev'].append(statistics.pstdev(sorted_vals))
            # self.history['pvariance'].append(statistics.pvariance(sorted_vals))
            # self.history['stdev'].append(statistics.stdev(sorted_vals))
            # self.history['variance'].append(statistics.variance(sorted_vals))
            # self.history['gen_mut_history'].append(np.mean(gen_mutation))

            # get surv bots
            npop_time = time.time()
            self.next_population = [el[0] for el in sorted_vals[:self.nsurv]]
            # self.next_population = [ self.population[vals.index(sorted_vals[i])] for i in range(self.nsurv) ]
            # print('npop___', time.time()-npop_time)
            # self.next_population = backend.create_next_population(self.nsurv, vals, sorted_vals, self.population)

            worst_bot = self.next_population[-1] # for printing

            # EXPENSIVE
            app_time = time.time()
            self.next_population += [self.__new_bot() for i in range(self.nnew)]
            # self.next_population += [backend.create_new_bot(
                # self.nnew, self.nsurv, self.nparents, self.bot_len, self.next_population, self.mut, self.weight_sample_enum
            # ) for i in range(self.nnew)]
            self.population = self.next_population
            # print('anpop__', time.time()-app_time)


            et = round(time.time()-start, 3)
            times.append(et)
            if verbose == 1:
                print(it, et, 
                    self.__bot2text(self.next_population[0]), self.__bot2text(worst_bot))
            
            if any(self._check_stoppings()):
                break
            
            # population mean mutations
#             self.history['mut_history'].append(np.mean(self.history['pop_mutations']))
#             self.history['pop_mutations'] = []
        print('mean generation time: {} sec'.format(np.mean(times)))


