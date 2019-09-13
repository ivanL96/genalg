from tkinter import *
import random


class VisualEnv:
    def __init__(self, root):
        self.canvas_width = 800
        self.canvas_height = 800
        self.circus_size = 10

        self.c = Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.c.pack()

    def draw_random_circ(self):
        x = random.randint(0, self.canvas_width-self.circus_size)
        y = random.randint(0, self.canvas_height-self.circus_size)    
        self.c.create_oval(x, y, x + self.circus_size, y + self.circus_size, width=2)


