import numpy as np
import random
from random import shuffle
import itertools

np.random.seed(0)
random.seed(0)


def init_population(n, n_weights, weights_range=1):
    population = np.random.rand(n, n_weights) * weights_range
    return population


def calculate_loss(population, loss_fn):
    """ Calculate loss from each pop row """
    return sorted(zip(loss_fn(population), population), key=lambda x: x[0], reverse=False)


def generate_next_pop(ranked_pop, n_weights, population, survived=10, parents=2, mutation=0.2):
    top_n = ranked_pop[:survived]
    surv_units = [u[1] for u in top_n]

    new_population = []
    for i in range(population - survived):
        shuffle(surv_units)
        _parents = surv_units[:parents]

        comb_parents = list(itertools.product(*_parents))  # all possible values combinations between parents
        # get n-th element for new unit
        comb_parents = [p+(random.random(),) for i,p in enumerate(comb_parents) if i%n_weights == 0]
        
        random_parent_element_idx = np.random.randint(parents+1, size=n_weights)  # random index mask
        new_unit = [par[idx] for idx, par in zip(random_parent_element_idx, comb_parents)]
        new_population.append(new_unit)
    return np.asarray(new_population + surv_units)


if __name__ == '__main__':
    def loss(pop):
        return np.sum(pop, axis=1)

    w = 10
    epochs = 1000
    p_size = 1000
    pop = init_population(p_size, w)

    for e in range(epochs):
        ranked_pop = calculate_loss(pop, loss)
        print(e,ranked_pop[0][0])
        newpop = generate_next_pop(ranked_pop, w, p_size)
        pop = newpop

    print(pop[0])
