#####################################################
##  TIPE main.py                                   ##
##  Passeron Paul, TUNEY Angelo MP                 ##
##  Adaptation de l'article "Cellular Model of     ##
##  Room Evacuation Based on Occupancy and         ##
##  Movement Prediction"                           ##
##  https://krbalek.cz/Science/ACRI_PH_MB_MK.pdf   ##
#####################################################

import math
import random

class Cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.n = 0
        self.potential = 0
        self.prediction = 0
        self.cell_type = 1  # 0: wall, barrier; 1: floor cell, exit
        self.predicted_direction = (0, 0)
        self.is_blocker_of = []
        self.bounded_to = None
        self.next_update = 0

    def get_probability(self, d):
        return 0


def get_reaction_surrounding():
    return [[Cell(-1, 1), Cell(0, 1), Cell(1, 1)], [Cell(-1, 0), None, Cell(1, 0)], [Cell(-1, -1), Cell(0, -1), Cell(1, -1)]]


class room():

    def __init__(self):
        self.total_n_of_persons = 10
        self.potential_strength = -3
        self.dimensions = (10, 10)
        self.exit = Cell()
        self.cells = {(i, j): Cell(i, j) for i in range(
            self.dimensions[0]) for j in range(self.dimensions[1])}
        self.N = 1
        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.time_unit = 0.7
        self.period = 1

    def add_cells(self, c1, c2):
        if c1 == None:
            return c2
        if c2 == None:
            return c1
        i1, j1 = c1.i, c1.j
        i2, j2 = c2.i, c2.j
        return self.cells[(i1+i2), (j1+j2)]

    def updateN(self):
        return 1

    def get_distance(self, x):
        return math.sqrt(10*(self.exit.i-x.i)**2/(self.exit.j-x.j)+(self.exit.j-x.j)**2)

    def get_potential(self, x):
        return -self.potential_strength*self.get_distance(x)

    def get_indicator(self, x, d):
        r = self.cells[self.add_cells(x, d)].r
        return r == int((d.i, d.j) == x.predicted_direction)

    def get_unnormalized_prob(self, x, d):
        target = self.cells[self.add_cells(x, d)]
        t = target.cell_type
        u = self.get_potential(target)
        n = target.n
        r_prime = self.get_indicator(x, d)
        return t * math.exp(self.alpha*u)*(1-self.beta*n)*(1-self.gamma*r_prime)

    def get_probabilities(self, x):
        dic = {(i, j): self.get_unnormalized_probs(x, Cell(i, j))
               for i in range(-1, 2) for j in range(-1, 2) if (i != j or i == 1)}
        inv_N = 0
        for k in dic:
            inv_N += dic[k]
        for k in dic:
            dic[k] /= inv_N
        return

    def get_agent_to_update(self):
        dic = {}
        for k in self.cells:
            c = self.cells[k]
            if c.n == 1:
                if c.next_update in dic:
                    dic[c.next_update].append(c)
                else:
                    dic[c.next_update] = [c]
        min_k = float("inf")
        for k in dic:
            if k < min_k:
                min_k = k
        return dic[min_k]

    def update_cells(self, cell_array):
        penalty = 1
        for c in cell_array:
            probabilities = self.get_probabilities(c)
            last_prob = 0
            dic_of_probs = {}
            for k in probabilities:
                last_prob+=probabilities[k]
                dic_of_probs[int(1000*last_prob)] = k
            rand_n = random.randint(0, 1000)
            closest_k = None
            for k in dic_of_probs:
                if closest_k == None:
                    closest_k = k
                elif math.abs(closest_k-rand_n) < math.abs(closest_k-rand_n):
                    closest_k = k
            target_dir = dic_of_probs[closest_k]
            target = self.cells[self.add_cells(c, target_dir)]
            c.next_update += self.period * penalty
