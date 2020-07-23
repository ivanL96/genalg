import numpy as np
import random
from random import shuffle
import itertools, time
from numba import njit, jit, vectorize

np.random.seed(0)
random.seed(0)


def init_population(n, n_weights, weights_range=1):
    population = np.random.rand(n, n_weights) * weights_range
    return population

def calculate_loss(population, loss_fn):
    """ Calculate loss from each pop row """
    return sorted(zip(loss_fn(population), population), key=lambda x: x[0], reverse=False)

@jit(nogil=True, nopython=True, parallel=True, cache=True)
def generate_next_pop(surv_units, n_weights, population_size, mutation=0.2):
    pop_weigths = n_weights * population_size
    repeats = pop_weigths // surv_units.size
    new_pop = np.repeat(surv_units, repeats)
    np.random.shuffle(new_pop)

    mutation_mask = np.full(pop_weigths, False)
    mutation_mask[:int(pop_weigths*mutation)] = True # mutate weight if true
    np.random.shuffle(mutation_mask)

    mut = np.random.rand(pop_weigths)
    new_pop[mutation_mask] = mut[mutation_mask]
    return new_pop.reshape((population_size, n_weights))

def get_top_n(ranked_pop, survived):
    top_n = ranked_pop[:survived]
    surv_units = [u[1] for u in top_n]
    surv_units = np.asarray(surv_units).flatten()
    return surv_units

if __name__ == '__main__':
    def loss(pop):
        _pop = pop#[:,:1]
        return np.sum(_pop, axis=1)

    w = 10
    p_size = 5000
    survived = 100
    mutation = 0.1
    pop = init_population(p_size, w)
    # warm up
    # generate_next_pop(np.random.rand(survived, w), w, p_size, mutation=mutation)
    start = time.time()
    epochs = 20000
    for e in range(epochs):
        ranked_pop = calculate_loss(pop, loss)
        print(e, ranked_pop[0][0])
        # min_loss = ranked_pop[0][0]
        # survived = survived+10 if e % 100 == 0 else survived
        surv_units = get_top_n(ranked_pop, survived=survived)
        newpop = generate_next_pop(surv_units, w, p_size, mutation=mutation)
        assert newpop.shape == pop.shape
        pop = newpop

    end = time.time()
    print(pop[0])
    print('total time',end-start)
