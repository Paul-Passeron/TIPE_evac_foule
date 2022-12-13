import math
import random
from nltk.probability import FreqDist, MLEProbDist
import matplotlib.pyplot as plt

class Cell:

    def __init__(self):
        self.type = 0  # 0 -> floor | 1 -> occupied | 2 -> wall | 3 -> exit
        self.waiting_list = []
        self.bound_to = []
        self.predicted_direction = (int(1), int(0))
        self.prediction = 0
        self.potential = 0
        self.next_update = 0.0

class Room:

    def __init__(self):
        self.potential_strength = 3
        self.dimX = 13
        self.dimY = 7
        self.cells = []
        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.mu = 1

    def initialize_cells(self):
        self.cells = [[Cell() for j in range(self.dimX)]
                      for i in range(self.dimY)]

    def closest_exit(self, i, j):
        min_d = float('inf')
        res = (0, 0)
        for x in range(self.dimX):
            for y in range(self.dimY):
                if self.cells[x][y].type == 3:
                    d = math.sqrt((x-i)**2+(y-j)**2)
                    if d < min_d:
                        min_d = d
                        res = (x, y)
        return res

    def get_distance(self, i, j):
        sortieX, sortieY = self.closest_exit(i, j)
        return math.sqrt((10*(sortieY-j)**2)/(sortieX-i)+(sortieX-i)**2)

    def get_potential(self, i, j):
        return -self.potential_strength*self.get_distance(i, j)

    def update_prediction(self):
        d = {}
        for i in range(self.dimX):
            for j in range(self.dimY):
                dX, dY = self.cells[i][j].predicted_direction
                if (i+dX, j+dY) in d:
                    d[(i+dX, j+dY)] += 1
                else:
                    d[(i+dX, j+dY)] = 1
        for i in range(self.dimX):
            for j in range(self.dimY):
                if (i, j) in d:
                    self.cells[i][j].prediction = d[i][j]
                else:
                    self.cells[i][j].prediction = 0

    def is_in_bounds(self, i, j):
        condition = i > 0 and i < self.dimX and j >= 0 and j < self.dimY
        return condition

    def get_indicator(self, i, j, dirX, dirY):
        if self.is_in_bounds(i+dirX, j+dirY) and self.cells[i][j].type == 1:
            return self.cells[i+dirX][j+dirY].prediction - int((dirX, dirY) == (self.cells[i][j].predicted_direction))
        return 0

    def get_unnormalized_prob(self, i, j, dirX, dirY):
        if self.cells[i][j].type != 1:
            return 0
        indexX, indexY = i+dirX, j+dirY
        if self.is_in_bounds(indexX, indexY):
            t = int(self.cells[indexX][indexY].type ==
                    0 or self.cells[indexX][indexY].type == 0)
            n = int(self.cells[indexX][indexY].type == 1)
            u = self.get_potential(indexX, indexY)
            r_prime_tilde = self.get_indicator(i, j, dirX, dirY)
            res = t * math.exp(self.alpha*u)*(1-self.beta*n) * \
                (1-self.gamma*r_prime_tilde)
            return res
        return 0

    def prob_condition(self, a, b, i, j):
        res = (a != b or a != 0 and self.get_unnormalized_prob(i, j, a, b) > 0)
        res = res and self.is_in_bounds(
            i+a, j+b) and self.cells[i+a][j+b].type != 2
        return res

    def get_probabilities(self, i, j):
        dic = {(a, b): self.get_unnormalized_prob(i, j, a, b)
               for a in range(-1, 2) for b in range(-1, 2) if self.prob_condition(a, b, i, j)}
        freq_dist = FreqDist(dic)
        prob_dist = MLEProbDist(freq_dist)
        return {c: prob_dist.prob(c) for c in dic}

    def choose_dir(self, i, j):
        dic = self.get_probabilities(i, j)
        if dic == {}:
            return (0, 0)
        return random.choices(list(dic.keys()), weights=list(dic.values()), k=1)[0]

    def get_cells_bound_to(self, i, j):
        L = []
        for x in range(self.dimX):
            for y in range(self.dimY):
                if self.cells[x][y].type == 1 and self.cells[x][y].bound_to == (i, j):
                    L.append((x, y))
        return L

    def get_agent_to_update(self):
        dic = {}
        for i in range(self.dimX):
            for j in range(self.dimY):
                if self.cells[i][j].type == 1:
                    nu = self.cells[i][j].next_update
                    if nu in dic:
                        dic[nu].append((i, j))
                    else:
                        dic[nu] = [(i, j)]
        sorted_keys = list(dic.keys())
        sorted_keys.sort()
        if sorted_keys != []:
            return dic[sorted_keys[0]]
        return []

    def move_agent(self, i1, j1, i2, j2):
        self.cells[i1][j1].bound_to = []
        self.cells[i1][j1].predicted_direction = (1, 0)
        self.cells[i1][j1].prediction = 0
        self.cells[i1][j1].potential = 0
        self.cells[i1][j1].next_update = 0
        self.cells[i1][j1].type = 0

        self.cells[i2][j2].bound_to = []
        self.cells[i2][j2].predicted_direction = (1, 0)
        self.cells[i2][j2].prediction = 0
        self.cells[i2][j2].potential = 0
        self.cells[i2][j2].next_update += 1+0.5*int(abs(i1-i2)+abs(j1-j2) > 1)
        self.cells[i2][j2].type = 1

        # We also have to unbound everyone that was bound to the
        # cell that was in i1, j1

    def unbound_cells(self, i, j):
        bound_cells = self.get_cells_bound_to(i, j)
        for (a, b) in bound_cells:
            self.cells[a][b].bound_to = []

    def resolve_conflict(self, i, j, l, depth=0):
        max_depth = 4
        if depth >= max_depth:
            return True
        rand_float = random.random()
        if len(l) > 1:
            if rand_float > self.mu:
                # One agent is selected to move
                sX, sY = random.choice(l)
                self.move_agent(sX, sY, i, j)
        elif len(l) == 1:
            # No conflict, the agent moves to the empty cell.
            a, b = l[0]
            self.move_agent(i, j, a, b)
            self.resolve_conflict(i, j, self.get_cells_bound_to(i, j), depth+1)
            # maybe unbound here
            self.unbound_cells(i, j)  # So is max recursion depth 1 ???
        return True

    def update_cells(self):
        # decision process
        cells_to_update = self.get_agent_to_update()
        target_cells = []
        self.update_prediction()
        for (i, j) in cells_to_update:
            target_dirX, target_dirY = self.choose_dir(i, j)
            if (target_dirX, target_dirY) != (0, 0):
                a, b = i + target_dirX, j + target_dirY
                self.cells[i][j].predicted_direction = (
                    target_dirX, target_dirY)
                if self.cells[i][j].type in (0, 3):
                    target_cells.append((a, b))
                    self.cells[i][j].waiting_list.append((i, j))
                elif self.cells[i][j].type == 1:
                    target_cells.append((a, b))
                    self.cells[i][j].bound_to = [(a, b)]
        for (i, j) in target_cells:
            w_l = self.cells[i][j].waiting_list
            self.resolve_conflict(i, j, w_l)


###TESTING

def get_array_to_display(piece):
    return [[3-piece.cells[i][j].type for i in range(piece.dimX)]for j in range(piece.dimY)]

def populate(piece, n):
    for _ in range(n):
        i = random.randint(0, math.floor(piece.dimX/2)-1)
        j = random.randint(0, piece.dimY-1)
        while piece.cells[i][j].type != 0:
            i = random.randint(0, math.floor(piece.dimX/2)-1)
            j = random.randint(0, piece.dimY-1)
        piece.cells[i][j].type = 1
    for i in range(piece.dimY):
        if piece.cells[-1][i].type == 0:
            piece.cells[-1][i].type = 1

def test_model(iter, n):
    piece = Room()
    plt.figure()
    plt.show()
    populate(piece, n)
    arr = []
    for _ in iter:
        arr = get_array_to_display(piece)
        piece.update_cells()
        plt.imshow(arr, cmap = 'binary')
    
