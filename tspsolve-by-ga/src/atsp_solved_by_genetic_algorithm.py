import time

from src import read_data
import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

DATA = None
CITY_NODE = None
POPULATION_SIZE = 50
GA_CHOOSE_RATION = 0.1
MUTATE_RATIO = 0.01
MAX_ITERATION = 100

class City():
    def __init__(self, name):
        self.name = int(name)

    def distance(self, city):
        distance = DATA[self.name][city.name]
        return distance

    def __repr__(self):
        return str(self.name+1)

class Fitness():
    def __init__(self, populations):
        self.populations = populations
        self.distance = 0
        self.fitness = 0.0

    def calculator_distance(self):
        if self.distance ==0:
            ruta_distance = 0
            for i in range(0, len(self.populations)):
                from_city = self.populations[i]
                to_city = None
                if i + 1 < len(self.populations):
                    to_city = self.populations[i + 1]
                else:
                    to_city = self.populations[0]
                ruta_distance += from_city.distance(to_city)
            self.distance = ruta_distance
        return self.distance

    def calculator_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.calculator_distance())
        return self.fitness

class atspSolutionByGeneticAlgorithm():
    def __init__(self,city_node, population_size, iteration):
        self.city_node = city_node
        self.population_size = population_size
        self.elite = (int)(GA_CHOOSE_RATION*population_size)
        self.mutate_ratio = MUTATE_RATIO
        self.iteration = iteration

        self.populations = self.inital_population(population_size, city_node)


    def inital_population(self,population_size,city_node):
        population = []
        for i in range(0, population_size):
            population.append(random.sample(city_node, len(city_node)))
        return population

    def gen_selection(self, sorted_city_fitness, elite):
        chosen_city = []
        df = pd.DataFrame(np.array(sorted_city_fitness), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

        for i in range(0, elite):
            chosen_city.append(sorted_city_fitness[i][0])

        for i in range(0, len(sorted_city_fitness) - elite):
            selection = 100*random.random()
            for i in range(0, len(sorted_city_fitness)):
                if selection <= df.iat[i,3]:
                    chosen_city.append(sorted_city_fitness[i][0])
                    break
        return chosen_city

    def compute_fitness(self,populations):
        city_fitness = {}
        for i in range(0,len(populations)):
            city_fitness[i] = Fitness(populations[i]).calculator_fitness()
        return sorted(city_fitness.items(), key = operator.itemgetter(1), reverse = True)

    def genetric_cross(self, parents, chosen_parents):
        cross = []
        for i in range(0, len(chosen_parents)):
            index = chosen_parents[i]
            cross.append(parents[index])
        return cross

    def mutaction(self,individal, mutation):
        for cru in range(len(individal)):
            if(random.random() < mutation):
                cru2 = int(random.random() * len(individal))

                city_node1 = individal[cru]
                city_node2 = individal[cru2]

                individal[cru] = city_node2
                individal[cru2] = city_node1
        return individal

    def gen_mutation(self, new_parents, mutation):
        mutation_parents = []

        for ind in range(0, len(new_parents)):
            individual_mutate = self.mutaction(new_parents[ind], mutation)
            mutation_parents.append(individual_mutate)
        return mutation_parents

    def descendent(self, parent_a, parent_b):
        new_parent_a = []
        gen_padre_a = int(random.random() * len(parent_a))
        gen_padre_b = int(random.random() * len(parent_b))

        gen_inicio = min(gen_padre_a, gen_padre_b)
        gen_termino = max(gen_padre_a, gen_padre_b)

        for i in range(gen_inicio, gen_termino):
            new_parent_a.append(parent_a[i])
        new_parent_b = [item for item in parent_b if item not in new_parent_a]

        new_parent = new_parent_a + new_parent_b
        return new_parent

    def generate_new_son(self, cross_parents, elite):
        new_sons = []
        length = len(cross_parents) - elite
        pool = random.sample(cross_parents, len(cross_parents))
        for i in range(0, elite):
            new_sons.append(cross_parents[i])

        for i in range(0, length):
            new_son = self.descendent(pool[i], pool[len(cross_parents)-i-1])
            new_sons.append(new_son)
        return new_sons

    def generacion_descendiente(self,parents, elite, mutation):
        # compute fitness
        sorted_city_fitness = self.compute_fitness(parents)
        # selection
        chosen_parents = self.gen_selection(sorted_city_fitness, elite)
        # cross
        cross_parents = self.genetric_cross(parents, chosen_parents)
        # new generations
        descendent_parents = self.generate_new_son(cross_parents, elite)
        # mutation
        new_parents = self.gen_mutation(descendent_parents, mutation)
        return new_parents

    def run(self):
        populations = self.populations
        print("Distance initial: " + str(1 / self.compute_fitness(populations)[0][1]))

        progress = []
        progress.append(1 / self.compute_fitness(populations)[0][1])
        for i in range(0, self.iteration):
            populations = self.generacion_descendiente(populations, self.elite, self.mutate_ratio)
            progress.append(1/self.compute_fitness(populations)[0][1])

        print("Distance Final (Solution): " + str(1 / self.compute_fitness(populations)[0][1]))
        best_fitness_index = self.compute_fitness(populations)[0][0]
        best_city_path = populations[best_fitness_index]
        print("best_city_path:" + str(best_city_path))

        plt.plot(progress)
        plt.ylabel('Distance')
        plt.xlabel('iteration')
        plt.show()
        return best_city_path

if __name__ == '__main__':
    data_path = read_data.read_atsp_path('data/atsp_17.txt')
    data = pd.read_csv(data_path,delim_whitespace=True, header = None)
    DATA = data
    list_city_node = []
    for j in range(0,len(data)):
        list_city_node.append(City(name=j))
    CITY_NODE = list_city_node
    start_time = time.time()
    mode = atspSolutionByGeneticAlgorithm(city_node=CITY_NODE, population_size=POPULATION_SIZE, iteration=MAX_ITERATION)
    mode.run()
    end_time = time.time()
    print('time_consumption {:.5f} s'.format(end_time-start_time))