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
import matplotlib.pyplot as plt


class Cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.n = 0
        self.potential = 0
        self.prediction = 0
        self.predicted_direction = (1, 0)
        self.is_blocker_of = []
        self.bound_to = []
        self.next_update = 0.0
        self.waiting_list = []

    def get_probability(self, d):
        return 0


def get_reaction_surrounding():
    return [[Cell(-1, 1), Cell(0, 1), Cell(1, 1)], [Cell(-1, 0), None, Cell(1, 0)], [Cell(-1, -1), Cell(0, -1), Cell(1, -1)]]


class room():

    def __init__(self):
        self.total_n_of_persons = 10
        self.potential_strength = 3
        self.dimensions = (13, 7)
        self.sortie = (self.dimensions[0]-1, int(self.dimensions[1]/2))
        self.exit = Cell(self.sortie[0], self.sortie[1])
        self.cells = {(i, j): Cell(i, j) for i in range(
            self.dimensions[0]) for j in range(self.dimensions[1])}
        self.cells[self.sortie] = self.exit
        self.N = 1
        self.alpha = 1#Pour le moment système homogène mais peut changer.
        self.beta = 0.5
        self.gamma = 0.9
        self.mu = 0.5
        self.time_unit = 0.7
        self.period = 1

    def add_cells(self, c1, c2):
        if c1 == None:
            return (c2.i, c2.j)
        if c2 == None:
            return (c1.i, c1.j)
        i1, j1 = c1.i, c1.j
        i2, j2 = c2.i, c2.j
        return ((i1+i2), (j1+j2))

    def updateN(self):
        return 1

    def get_distance(self, x):
        if (x.i == self.exit.i):
            return 0
        return math.sqrt(10*(self.exit.j-x.j)**2/(self.exit.i-x.i)+(self.exit.i-x.i)**2)
    def get_potential(self, x):
        return -self.potential_strength*self.get_distance(x)
    def get_indicator(self, x, d):
        r = self.cells[self.add_cells(x, d)].prediction
        return r == int((d.i, d.j) == x.predicted_direction)

    def get_unnormalized_prob(self, x, d):
        if x.i != 12:
            index = self.add_cells(x, d)
            if index in self.cells.keys():
                target = self.cells[index]
                t = 1
                if target.n not in (0, 3):
                    t = 0
                u = self.get_potential(target)
                n = target.n
                if n > 1:
                    n = 0
                r_prime = self.get_indicator(x, d)
                return t * math.exp(self.alpha*u)*(1-self.beta*n)*(1-self.gamma*r_prime)
            else:
                return 0
        else:
            return 0

    def get_probabilities(self, x):
      
        dic = {(i, j): self.get_unnormalized_prob(x, Cell(i, j))
               for i in range(-1, 2) for j in range(-1, 2) if (i != j or i != 0 and self.get_unnormalized_prob(x, Cell(i, j)) > 0)}
        inv_N = 0
        for k in dic:
            inv_N += dic[k]
        if inv_N == 0:
            return {(0, 0): 1}
        for k in dic:
            dic[k] /= inv_N
        return dic

    def get_agent_to_update(self):
        # Selection of active agents
        dic = {}
        for k in self.cells:
            c = self.cells[k]
            if c.n == 1:
                if c.next_update in dic:
                    dic[c.next_update].append((c.i, c.j))
                else:
                    dic[c.next_update] = [(c.i, c.j)]
        min_k = float("inf")
        for k in dic:
            if k < min_k:
                min_k = k
        if min_k == float("inf"):
            return []
        return dic[min_k]

    def choose_dir(self, c):
        if abs(c.i - self.exit.i) == 1 and (c.j - self.exit.j) >= 1:
            return Cell(0, -1)
        if abs(c.i - self.exit.i) == 1 and (c.j - self.exit.j) <= -1:
            return Cell(0, 1)
        if abs(c.i - self.exit.i) == 1 and (c.j - self.exit.j) == 0:
            return Cell(1, 0)
        prob_res = 1000
        probabilities = self.get_probabilities(c)
        last_prob = 0
        dic_of_probs = {}
        for k in probabilities:
            last_prob += probabilities[k]
            dic_of_probs[int(prob_res*last_prob)] = k
        rand_n = random.randint(0, prob_res)
        closest_k = None
        for k in dic_of_probs:
            if closest_k == None:
                closest_k = k
            elif abs(k-rand_n) < abs(closest_k-rand_n):
                closest_k = k
        a, b = dic_of_probs[closest_k]
        return Cell(a, b)

    def get_cells_bound_to(self, c):
        L = []
        for i, j in self.cells:
            cell = self.cells[(i, j)]
            if i != c.i and j != c.j:
                if cell.bound_to == (c.i, c.j):
                    L.append((i, j))
        return L
    
    def resolve_conflict(self, i, j, w_l):
        penalty = 1
        current_cell = self.cells[(i, j)]
        if current_cell.n in (0, 3):
            #w_l = current_cell.waiting_list
            if len(w_l) == 1:
                #move cell because is alone in waiting list of empty cell
                ag = self.cells[w_l[0]]
                if self.cells[w_l[0]].n == 1:
                    self.cells[w_l[0]].n = 0
                    if (i, j) != self.sortie:
                        self.cells[(i, j)].n = 1
                        if abs(ag.i - i) + abs(ag.j - j) >= 2:
                            penalty = 3/2
                        self.cells[(i, j)].next_update = ag.next_update+self.period*penalty
            elif len(w_l) > 1:
                proba = random.randint(0, 1000)
                if proba <= self.mu * 1000:
                    #disable all agents movement
                    pass
                else:
                    #choose one random agent to move
                    random_agent_index = w_l[random.randint(0, len(w_l)-1)]
                    random_agent = self.cells[random_agent_index]
                    if self.cells[random_agent_index].n == 1:
                        self.cells[random_agent_index].n = 0
                        if abs(random_agent.i - i) + abs(random_agent.j - j) >= 2:
                            penalty = 3/2
                        self.cells[(i, j)].next_update = random_agent.next_update+self.period*penalty
                        if (i, j) != self.sortie:
                            self.cells[(i, j)].n = 1
                    #Implementing 3/2 penalty for diagonal movement
                    
                    #random_agent moved so we have to unbound every cell that
                    #was bound to random_agent.
                    #Bound agent also have to move after the blocker moved but
                    #will be implemented later.
                    
                    
                    if random_agent.is_blocker_of != []:
                        ag = random_agent.is_blocker_of[0]
                        random_agent.is_blocker_of = random_agent.is_blocker_of[1:]
                        if self.cells[ag].n == 1:
                            self.cells[ag].n = 0
                            self.cells[random_agent_index].n = 1
                            self.cells[random_agent_index].next_update = self.cells[ag].next_update + self.period * penalty
                    
                    
                    for cell_to_unbound_index in random_agent.is_blocker_of:
                        if cell_to_unbound_index in self.cells.keys():
                            self.cells[cell_to_unbound_index].bound_to = []
                    if random_agent_index in self.cells.keys():
                        self.cells[random_agent_index].is_blocker_of = []
                        #Resetting waiting list of random_agent cell
                        self.cells[random_agent_index].waiting_list = []
    
    def update_cells(self):
        cell_array = self.get_agent_to_update()
        # Decision process
        penalty = 1
        for (i, j) in cell_array:
            c = self.cells[(i, j)]
            # unbounding cellls bound to c as c is updating
            cells_bound_to = self.get_cells_bound_to(c)
            for (ci, cj) in cells_bound_to:
                self.cells[(ci, cj)].bound_to = []
            target_dir = self.choose_dir(c)
            self.cells[(i, j)].predicted_direction = (target_dir.i, target_dir.j)
            a, b = self.add_cells(c, target_dir)
            
            if (a, b) in self.cells.keys():
                if self.cells[(a, b)].n in (0, 3):
                    self.cells[(a, b)].waiting_list.append((i, j))
                else:
                    self.cells[(a, b)].is_blocker_of.append((i, j))
                    self.cells[(i, j)].bound_to = [(a, b)]
        
        #Conflict solution and motion
        for (i, j) in self.cells:
            w_l = self.cells[(i, j)].waiting_list
            self.resolve_conflict(i, j, w_l)
            #self.resolve_conflict(i, j, w_l_2)
            # current_cell = self.cells[(i, j)]
            # #case 3)a): Targetted cell is empty
            # if current_cell.n in (0, 3):
            #     w_l = current_cell.waiting_list
            #     if len(w_l) == 1:
            #         #move cell because is alone in waiting list of empty cell
            #         if self.cells[w_l[0]].n == 1:
            #             self.cells[w_l[0]].n = 0
            #             if (i, j) != self.exit:
            #                 self.cells[(i, j)].n = 1
            #     elif len(w_l) > 1:
            #         proba = random.randint(0, 1000)
            #         if proba <= self.mu * 1000:
            #             #disable all agents movement
            #             pass
            #         else:
            #             #choose one random agent to move
            #             random_agent_index = w_l[random.randint(0, len(w_l)-1)]
            #             random_agent = self.cells[random_agent_index]
            #             if self.cells[random_agent_index].n == 1:
            #                 self.cells[random_agent_index].n = 0
            #                 print((i, j))
            #                 if (i, j) != self.exit:
            #                     self.cells[(i, j)].n = 1
            #             #Implementing 3/2 penalty for diagonal movement
            #             if abs(random_agent.i - i) + abs(random_agent.j - j) >= 2:
            #                 penalty = 3/2
            #             #random_agent moved so we have to unbound every cell that
            #             #was bound to random_agent.
            #             #Bound agent also have to move after the blocker moved but
            #             #will be implemented later. 
            #             for cell_to_unbound_index in random_agent.is_blocker_of:
            #                 if cell_to_unbound_index in self.cells.keys():
            #                     self.cells[cell_to_unbound_index].bound_to = []
            #             if random_agent_index in self.cells.keys():
            #                 self.cells[random_agent_index].is_blocker_of = []
            #                 #Resetting waiting list of random_agent cell
            #                 self.cells[random_agent_index].waiting_list = []
            #case 3)b) Targetted cell isn'tempty we will implement 3)b later
            # else:
            #     pass

            
            for a in range(self.dimensions[1]):
                self.cells[(self.dimensions[0]-1, a)].n = 2
            self.cells[self.sortie].n = 3
            # self.cells[(i, j)].next_update += self.period * penalty
            
        for a in range(self.dimensions[1]):
            self.cells[(self.dimensions[0]-1, a)].n = 2
        self.cells[self.sortie].n = 3
def populate(room, n):
    room.total_n_of_persons = n
    for _ in range(n):
        i = random.randint(0, int(room.dimensions[0]/2)-1)
        j = random.randint(0, room.dimensions[1]-1)
        while room.cells[(i, j)].n == 1 and (i, j) != room.exit:
            i = random.randint(0, int(room.dimensions[0]/2)-1)
            j = random.randint(0, room.dimensions[1]-1)
        room.cells[(i, j)].n = 1
    for i in range(room.dimensions[1]):
        room.cells[(room.dimensions[0]-1, i)].n = 2


def get_array_to_display(room):
    return [[(room.cells[(i, j)].n==0) for i in range(room.dimensions[0])]for j in range(room.dimensions[1])]

## TESTING THE MODEL
piece = room()
populate(piece, 10)
piece.cells[piece.sortie].n = 3
arr = get_array_to_display(piece)
result = []
prev_arr = arr
#plt.imshow(arr, cmap = 'binary', )


for _ in range(1000):
    cter = 0
    for c in piece.cells:
        if piece.cells[c].n == 1:
            cter+=1
    #plt.figure()
    arr = get_array_to_display(piece)
    
        
    result.append(arr)
    
    #plt.imshow(arr, cmap = 'binary')
    piece.update_cells()
    # plt.pause(0.01)
    # print(cter)
    if cter == 0:
        break
    prev_arr = arr


plt.figure()
for arr in result:
    plt.imshow(arr, cmap = 'binary')
    plt.show()
