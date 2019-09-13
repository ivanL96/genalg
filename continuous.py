import numpy as np
import random
import time
from tkinter import *
from visual import *


class ContinuousModel:
    def __init__(self, nbots, nparents, mut, bot_len=None, bot_len_range=(0, 10)):
        self.nbots = nbots
        self.bot_len = bot_len
        self.bot_len_range = range(*bot_len_range)
        self.nparents = nparents
        self.mut = mut
    
    def random_bot(self, length=None):
        if length is not None:
            len_range = range(length)
        else:
            len_range = range(self.bot_len) if self.bot_len else self.bot_len_range
        return [random.random() for el in len_range]

    def create_population(self, canv):
        for i in range(self.nbots):
            rand_len = random.randint(1, 10)
            bot = self.random_bot(length=rand_len)
            canv.draw_random_circ()
            yield bot

    def run(self, canv):
        bots = list(self.create_population(canv))
        # print(bots)



if __name__ == '__main__':
    root = Tk()
    canv = VisualEnv(root)
    ContinuousModel(100, 2, 0.2).run(canv)
    root.mainloop()
