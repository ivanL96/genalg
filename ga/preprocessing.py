import numpy as np


def to_ohe(sequence):
    elements = list(set(sequence))
    return [elements.index(el) for el in sequence]


def normalize(data):
    data = np.asarray(data)
    return (data - min(data)) / (max(data) - min(data))


def bot_to_env_interpreter():
    pass