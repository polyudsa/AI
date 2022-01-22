import random
import math
import time

import numpy as np
import matplotlib.pyplot as plt
import read_data

GA_CHOOSE_RATION = 0.1
MUTATE_RATIO = 0.9
POPULATION_SIZE = 100
MAX_ITERATION = 1000

class tspSolutionByGeneticAlgorithm():

    def __init__(self, num_city, population_size=50, iteration=100, data=None):
        # the number of cities
        self.num_city = num_city
        # population size
        self.population_size = population_size
        # fitness
        self.fitness = []
        # number of generations
        self.iteration = iteration
        # data [[1, 0.123, 0.231],[...]]
        self.location = data
        # probability of cross cover
        self.ga_choose_ratio = GA_CHOOSE_RATION
        # probability of mutation
        self.mutate_ratio = MUTATE_RATIO
        # the distance of cities [[inf,1,1,1,1][1,inf,1,1,1].....]
        self.distance_cities = self.compute_distance_cities(num_city, data)
        # Initial population [[1,2,3,4,5],[2,3,1,4,5],.....]
        self.population = self.inital_population(self.distance_cities, population_size, num_city)
        # Show the best path after initialization
        fitness_score = self.compute_fitness(self.population)
        # Sort in descending order and return index
        sort_index = np.argsort(-fitness_score)

        # Store the results of each iteration and draw the convergence graph
        self.iter_x = [0]
        self.iter_y = [1. / fitness_score[sort_index[0]]]

    def inital_population(self, distance_cities, population_size, num_city):
        start_index = 0
        result_best_path = []
        for i in range(population_size):
            rest = [x for x in range(0, num_city)]
            # All starting points have been generated
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result_best_path.append(result_best_path[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # Find a nearest neighbor path
            result_best_path_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if distance_cities[current][x] < tmp_min:
                        tmp_min = distance_cities[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_best_path_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result_best_path.append(result_best_path_one)
            start_index += 1
        return result_best_path
    # Calculate the distance between different cities
    def compute_distance_cities(self, num_city, location):
        # build a zero array (num_city*num_city) [0,0,...]
        print(location)
        distance_cities = np.zeros((num_city, num_city))

        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    distance_cities[i][j] = np.inf
                    continue
                x = location[i]
                y = location[j]
                # calculator Euler distance
                tmp_distance = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(x, y)]))
                distance_cities[i][j] = tmp_distance
        return distance_cities

    # Calculate path length
    def compute_path_len(self, best_paths, distance_cities):
        result = 0
        try:
            the_first_path = best_paths[0]
            the_last_path = best_paths[-1]
            result = distance_cities[the_first_path][the_last_path]
        except:
            import pdb
            pdb.set_trace()
        for i in range(len(best_paths) - 1):
            the_first_path = best_paths[i]
            the_last_path = best_paths[i + 1]
            result += distance_cities[the_first_path][the_last_path]
        return result

    # Calculate population fitness
    def compute_fitness(self, populations):
        fitness = []
        for population in populations:
            if isinstance(population, int):
                import pdb
                pdb.set_trace()
            length = self.compute_path_len(population, self.distance_cities)
            fitness.append(1.0 / length)
        return np.array(fitness)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def genetric_cross(self, parent_x, parent_y):
        len_ = len(parent_x)
        assert len(parent_x) == len(parent_y)
        path_list = [t for t in range(len_)]
        order_list = list(random.sample(path_list, 2))
        order_list.sort()
        start, end = order_list

        # Find the conflict points and save their subscripts,
        # x stores the subscripts in y,
        # and y stores the subscripts that x and it conflict with
        tmp = parent_x[start:end]
        x_conflict_index = []
        for sub in tmp:
            index = parent_y.index(sub)
            if not (index >= start and index < end):
                x_conflict_index.append(index)

        y_confict_index = []
        tmp = parent_y[start:end]
        for sub in tmp:
            index = parent_x.index(sub)
            if not (index >= start and index < end):
                y_confict_index.append(index)

        assert len(x_conflict_index) == len(y_confict_index)

        # cross
        tmp = parent_x[start:end].copy()
        parent_x[start:end] = parent_y[start:end]
        parent_y[start:end] = tmp

        # Conflict resolution
        for index in range(len(x_conflict_index)):
            i = x_conflict_index[index]
            j = y_confict_index[index]
            parent_y[i], parent_x[j] = parent_x[j], parent_y[i]

        assert len(set(parent_x)) == len_ and len(set(parent_y)) == len_
        return list(parent_x), list(parent_y)

    def genetic_parent(self, fitness_scores, ga_choose_ratio):
        sort_index = np.argsort(-fitness_scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_fitness_score = []
        for index in sort_index:
            parents.append(self.population[index])
            parents_fitness_score.append(fitness_scores[index])
        return parents, parents_fitness_score

    def genentic_choose(self, genes_score, genes_choose):
        index1 = index2 = 0
        sum_score = sum(genes_score)
        # The probability of 𝑥i being chosen is fitness)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def genetric_mutate(self, gene):
        path_list = [t for t in range(len(gene))]
        order_list = list(random.sample(path_list, 2))
        start, end = min(order_list), max(order_list)
        tmp = gene[start:end]
        # np.random.shuffle(tmp)
        tmp = tmp[::-1]
        gene[start:end] = tmp
        return list(gene)

    def genetic_algorithm(self):
        # Get quality parents
        fitness_scores = self.compute_fitness(self.population)
        # Select some outstanding individuals as the parent candidate set
        parents, parents_score = self.genetic_parent(fitness_scores, self.ga_choose_ratio)
        tmp_best_one = parents[0]
        tmp_best_score = parents_score[0]
        # New population
        new_population = parents.copy()
        # Generate a new population
        while len(new_population) < self.population_size:
            # Roulette way to choose the parent
            gene_x, gene_y = self.genentic_choose(parents_score, parents)
            # cross
            gene_x_new, gene_y_new = self.genetric_cross(gene_x, gene_y)
            # Mutations
            if np.random.rand() < self.mutate_ratio:
                gene_x_new = self.genetric_mutate(gene_x_new)
            if np.random.rand() < self.mutate_ratio:
                gene_y_new = self.genetric_mutate(gene_y_new)
            x_fitness = 1. / self.compute_path_len(gene_x_new, self.distance_cities)
            y_fitness = 1. / self.compute_path_len(gene_y_new, self.distance_cities)
            # Put the highly adaptable into the population
            if x_fitness > y_fitness and (not gene_x_new in new_population):
                new_population.append(gene_x_new)
            elif x_fitness <= y_fitness and (not gene_y_new in new_population):
                new_population.append(gene_y_new)

        self.population = new_population

        return tmp_best_one, tmp_best_score

    def run(self):
        best_path_list = None
        best_fitness_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.genetic_algorithm()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_fitness_score:
                best_fitness_score = tmp_best_score
                best_path_list = tmp_best_one
            self.best_record.append(1. / best_fitness_score)
            # plot
            # distance = self.compute_path_len(tmp_best_one,self.distance_cities)
            # tmp_best_one_xy = np.vstack([self.location[tmp_best_one], self.location[tmp_best_one][0]])
            # self.common_plot(distance,tmp_best_one_xy[:,0],tmp_best_one_xy[:,1],False)
            # print("iteration best one path: ",tmp_best_one)
            # print("{} iteration best one path fitness score: {}".format(i,tmp_best_score))
            # plt.pause(0.01)

        best_path, best_score = self.location[best_path_list], 1. / best_fitness_score
        best_distance = self.compute_path_len(best_path_list,self.distance_cities)
        best_path = np.vstack([best_path, best_path[0]])
        self.common_plot(best_distance,best_path[:, 0], best_path[:, 1],True,best_path_list)
        result_best_path_list = []
        for i in best_path_list:
            result_best_path_list.append(i+1)
        print("the best path: ",result_best_path_list)
        print("the best score: ",best_score)
        return best_path, best_score

    def common_plot(self, distance,x,y,show_iterator,best_path_list):
        plt.clf()
        fig, axs = plt.subplots(2, 1)
        axs[0].scatter(x, y)
        axs[0].plot(x, y)
        axs[0].set_title(str('planning result cost: {}').format(distance))
        j = 0
        for i in best_path_list:
            axs[0].text(x[j], y[j], i+1,
                    fontsize=10, color = "r", style = "italic", weight = "light",
                    verticalalignment='bottom', horizontalalignment='right',rotation=10)
            j += 1
        if show_iterator:
            iterations = range(self.iteration)
            best_record = self.best_record
            axs[1].plot(iterations, best_record,'r')
            axs[1].set_title('')
        plt.show()

if __name__ == '__main__':
    # 10 cities
    data = read_data.read_tsp('data/ch30.tsp')
    data = np.array(data)
    data = data[:, 1:]
    start_time = time.time()
    mode = tspSolutionByGeneticAlgorithm(num_city=data.shape[0], population_size=POPULATION_SIZE, iteration=MAX_ITERATION, data=data.copy())
    # mode.run()
    end_time = time.time()
    print('time_consumption {:.5f} s'.format(end_time-start_time))
    # 20 cities
    # calculator('data/ch20.tsp')
