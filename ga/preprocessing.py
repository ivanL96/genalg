import numpy as np
from random import sample


def to_ohe(sequence):
    elements = list(set(sequence))
    return [elements.index(el) for el in sequence]


def normalize(data):
    data = np.asarray(data)
    return (data - min(data)) / (max(data) - min(data))

def randomize_string(string):
    return ''.join(sample(string, len(string) ))


def bot_to_env_interpreter():
    pass