import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import datetime as dt
from datetime import timedelta, datetime

from src import read_data

DISTANCES = None
READY_TIME = None
COORDINATE = None
DURATION_TIME = []
POPULATION_SIZE = 0
GA_CHOOSE_RATION = 0.1
MUTATE_RATIO = 0.01
MAX_ITERATION = 1000

class tsptwSolutionByGeneticAlgorithm():
    """
    start_city: 起始城市下标索引
    destination: 目标城市索引列表
    visit_duration: 目标城市拜访时长列表
    time_user: 用户起始拜访时间
    """
    def __init__(self,start_city,destination,visit_duration,time_user, pop_size, crossover_rate,mutation_rate,iteration):
        self.start_city = start_city
        self.destination = destination
        self.visit_duration = visit_duration
        self.time_user = datetime.strptime(time_user, '%H:%M').time()
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.iteration = iteration
        self.offspring_dist_result = None
        self.offspring_total_dist = None
        self.offspring_result = None
        self.visit_time_result = None
        self.time_offspring_result = None
        self.crossover_result = None
        self.mutation_result = None
        self.penalty_0_result = None
        self.penalty_i_result = None
        self.penalty_result = None
        self.fitness_result = None
        self.evaluate_best = None
        self.evaluate_selected = None
        self.chromosome_result = None
        self.penalty_i_arrived = None
        self.penalty_i_start = None
        self.penalty_0_arrived_time = None
        self.schedule_result = None
        self.s_user = None


    def chromosome(self,destination,pop_size):
        chrom = []
        for i in range (pop_size):
            destination = shuffle(destination)
            chrom.append(destination)
        self.chromosome_result = np.array(chrom)
        return self.chromosome_result

    def choosing_parent(self,chromosome, offspring):
        parent = random.sample(range(0, chromosome.shape[0]), offspring)
        return chromosome[parent]

    def crossover(self,chromosome, crossover_rate):
        pop_size = len(chromosome[0])
        offspring = (round(crossover_rate * pop_size) + 1)
        parents = self.choosing_parent(chromosome, offspring).tolist()
        cross_off1 = []
        cross_off2 = []
        for i in range(offspring - 1):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            firstCrossPoint = np.random.randint(0, len(parent1) - 2)
            secondCrossPoint = np.random.randint(firstCrossPoint + 1, len(parent1) - 1)

            parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
            parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

            temp_child1 = parent1[:firstCrossPoint] + parent2MiddleCross + parent1[secondCrossPoint:]
            temp_child2 = parent2[:firstCrossPoint] + parent1MiddleCross + parent2[secondCrossPoint:]

            relations = []
            for j in range(len(parent1MiddleCross)):
                relations.append([parent2MiddleCross[j], parent1MiddleCross[j]])

            child1 = self.recursion1(relations,parent1,temp_child1, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
            child2 = self.recursion1(relations,parent2,temp_child2, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)

            cross_off1.append(child1)
            cross_off2.append(child2)
        self.crossover_result = np.vstack((cross_off1, cross_off2))
        return self.crossover_result

    def recursion1(self,relations,parent1, tempChild, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross):
        child = np.array([0 for i in range(len(parent1))])
        for i,j in enumerate(tempChild[:firstCrossPoint]):
            c = 0
            for relation in relations:
                if j == relation[0]:
                    child[i] = relation[1]
                    c = 1
                    break

            if c == 0:
                child[i] = j

        j = 0
        for i in range(firstCrossPoint,secondCrossPoint):
            child[i] = parent2MiddleCross[j]
            j += 1

        for i,j in enumerate(tempChild[secondCrossPoint:]):
            c = 0
            for relation in relations:
                if j == relation[0]:
                    child[i + secondCrossPoint] = relation[1]
                    c = 1
                    break

            if c == 0:
                child[i + secondCrossPoint] = j

        childUnique = np.unique(child)
        if len(child) > len(childUnique):
            child = self.recursion1(relations,parent1,child, firstCrossPoint, secondCrossPoint, parent1MiddleCross, parent2MiddleCross)
        return(child)

    def mutation(self,chromosome,mutation_rate):
        pop_size = len(chromosome[0])
        offspring = (round(mutation_rate * pop_size))
        parents = self.choosing_parent(chromosome, offspring)
        rand = np.random.randint(0, len(chromosome[0]), 2)
        parents[:, [rand[0], rand[1]]] = parents[:, [rand[1], rand[0]]]
        child = parents
        self.mutation_result = np.array(child)
        return self.mutation_result

    def offspring(self,chromosome, crossover, mutation):
        # penggabungan offspring
        offsprings = np.vstack((chromosome, crossover, mutation))
        self.offspring_result = np.array(offsprings)
        return self.offspring_result

    def dist_offspring(self,s_user, offspring, distance):
        # melakukan transpose dari offspring
        off = np.transpose(offspring)
        result = []
        total = []
        total_dist = None
        for s in range(len(s_user)):
            # jarak awal
            total_dist = s_user[s]
            for i in range(len(off) - 1):
                dist = distance[(off[i]) - 1, (off[i + 1]) - 1]
                result.append(dist)
                # total jarak tiap kromosom
                total_dist += dist
        total.append(total_dist)
        # jarak dari lokasi ke lokasi
        self.offspring_dist_result = np.array(result)
        # total jarak
        self.offspring_total_dist = np.array(total)

        return self.offspring_dist_result, self.offspring_total_dist

    # mencari best time dari offspring
    def time_offspring(self, offspring, time):
        best_time = []
        for j in range(len(offspring)):
            time_kromosom = []
            for i in range(len(offspring[j])):
                bestTime = time[(offspring[j][i]) - 1]
                time_kromosom.append(bestTime)
            best_time.append(time_kromosom)
        self.time_offspring_result = np.array(best_time)
        return self.time_offspring_result

    def visit_time(self,destination, visit_duration, offspring):
        # relasi antara destinasi dan waktu kunjungannya
        relations = np.dstack((destination, visit_duration))
        relations = np.squeeze(relations)

        # generate waktu kunjungan dari offspring
        duration = np.array([[0 for col in range(len(offspring[0]))] for row in range(len(offspring))])
        for i in range(len(duration)):
            for j in range(len(duration[i])):
                findRow = np.where(relations == offspring[i, j])
                row = list(zip(findRow[0]))
                duration[i, j] = relations[row, 1]
        self.visit_time_result = duration
        return self.visit_time_result

    def penalty_0(self, s_user, time_user, offspring, distances, times):
        arrived_time = []

        # penalty awal
        penalty_0_result = []
        for s in range(len(s_user)):
            # waktu perjalanan dari lokasi user
            t_0 = (s_user[s]/60) * 60
            t_0 = timedelta(minutes = t_0)

            # menghitung waktu tiba ke lokasi pertama
            start_time_0 = (dt.datetime.combine(dt.date(1, 1, 1), time_user) + t_0).time()
            arrived_time.append(start_time_0)

            # mengambil best time lokasi pertama
            off_time = times[s , 0]
            if off_time > start_time_0:
                p_0 = dt.datetime.combine(dt.date(1, 1, 1), off_time) - dt.datetime.combine(dt.date(1, 1, 1), start_time_0)
            elif start_time_0 > off_time:
                p_0 = dt.datetime.combine(dt.date(1, 1, 1), start_time_0) - dt.datetime.combine(dt.date(1, 1, 1), off_time)
            p_float = p_0.total_seconds()
            penalty_0_result.append(p_float)
        self.penalty_0_arrived_time = arrived_time
        self.penalty_0_result = penalty_0_result

    def penalty_i(self,start_time_0, offspring, distances, times, visit_duration):
        off = np.transpose(offspring)
        times = np.transpose(times)
        visit_dur = np.transpose(visit_duration)
        arrived_time = start_time_0

        # menyimpan waktu kunjungan tiap lokasi
        arrived = []
        start = []
        arrived.append(arrived_time)

        # penalty i
        penalty_i = []
        for i in range(len(off) - 1):
            # distance
            dist = distances[(off[i]) - 1, (off[i + 1]) - 1]
            trip_time = (dist/60) * 60
            trip_time = trip_time * timedelta(minutes=1)

            # best time lokasi selanjutnya
            best_time = times[i + 1]

            # visit duration
            visit_duration = visit_dur[i]
            visit_duration = visit_duration * timedelta(minutes=1)

            # start time
            start_time = []
            for at in range(min(len(arrived_time), len(visit_duration))):
                start_time_i = (dt.datetime.combine(dt.date(1, 1, 1), arrived_time[at]) + visit_duration[at]).time()
                start_time.append(start_time_i)
            start.append(start_time)

            # arrived time
            arrived_time_i = []
            temp_penalty_i = []
            for index in range(min(len(start_time), len(trip_time), len(best_time))):
                arrived_time = (dt.datetime.combine(dt.date(1, 1, 1), start_time[index]) + trip_time[index]).time()
                arrived_time_i.append(arrived_time)
                # best time match
                if arrived_time > best_time[index]:
                    p_i = dt.datetime.combine(dt.date(1, 1, 1), arrived_time) - dt.datetime.combine(dt.date(1, 1, 1), best_time[index])
                elif best_time[index] > arrived_time:
                    p_i = dt.datetime.combine(dt.date(1, 1, 1), best_time[index]) - dt.datetime.combine(dt.date(1, 1, 1), arrived_time)
                p_i_float = p_i.total_seconds()
                temp_penalty_i.append(p_i_float)
            # penalty i
            penalty_i.append(temp_penalty_i)
            # replace arrived time to the new one
            arrived_time = arrived_time_i
            arrived.append(arrived_time)

        # last location
        lastArrived = arrived[len(arrived) - 1]
        lastVisit = visit_dur[len(visit_dur) - 1]
        lastVisit = lastVisit * timedelta(minutes=1)
        last_start = []
        for lastLoc in range(min(len(lastArrived), len(lastVisit))):
            last_start_time = (dt.datetime.combine(dt.date(1, 1, 1), lastArrived[lastLoc]) + lastVisit[lastLoc]).time()
            last_start.append(last_start_time)
        start.append(last_start)

        # total penalty i
        total = [sum(x) for x in zip(*penalty_i)]
        self.penalty_i_result = total
        self.penalty_i_start = start
        self.penalty_i_arrived = arrived


    def penalty(self, penalty_0, penalty_i):
        total_penalty = [x + y for x, y in zip(penalty_0, penalty_i)]
        self.penalty_result = total_penalty
        return self.penalty_result

    def fitness(self, distances, penalty):
        fitness = []
        total_fitnes = 0
        for off in range(len(penalty)):
            total_fitnes = 1/(distances + penalty)
        fitness.append(total_fitnes)
        self.fitness_result = np.array(fitness)
        return self.fitness_result

    def evaluate(self,offspring, fitness, popSize):
        fitness = np.transpose(fitness)
        offspring = offspring[:, :, np.newaxis]
        merg = np.hstack((offspring, fitness))
        fit_sort = sorted(merg, key=lambda col: -col[len(merg[0]) - 1])
        selected = np.array(fit_sort[0:popSize])
        selected = np.squeeze(selected)
        best = selected.astype(int)
        self.evaluate_selected = best[:, 0:len(selected[0]) - 1]
        self.evaluate_best = np.array(selected[0, 0:len(selected[0])])

    # waktu/jadwal tiap lokasi
    def schedule(self,offspring, arrived, start, best):
        arrived = np.transpose(arrived)
        start = np.transpose(start)
        temp = np.dstack((offspring, arrived, start))
        route = []
        for i in range(len(temp)):
            time = temp[i][:, 0]
            comparison = time == best
            if comparison.all():
                route = temp[i]
        self.schedule_result = route

    def distances(self,user_loc,offspring):
        s_user = []
        dest = offspring[:, 0]
        for destination in dest:
            dist = DISTANCES[user_loc][destination]
            s_user.append(dist)

        self.s_user = np.array(s_user)
        return self.s_user

    def run(self):
        destination = self.destination
        pop_size = self.pop_size
        epochs = self.iteration
        chromosome_result = self.chromosome(destination, pop_size)
        best_fitness = []
        self.best_record = []
        for epoch in range(epochs):
            # crossover
            crossover_result = self.crossover(chromosome_result, self.crossover_rate)

            # mutation
            mutation_result = self.mutation(chromosome_result, self.mutation_rate)

            # offspring
            offspring_result = self.offspring(chromosome_result, crossover_result, mutation_result)

            # user dist
            s_user = self.distances(self.start_city, offspring_result)

            # dist offspring
            distOffspring = self.dist_offspring(s_user, offspring_result, DISTANCES)

            # time offspring
            timeOffspring = self.time_offspring(offspring_result, READY_TIME)

            # visit duration
            visit_duration = self.visit_time(destination, self.visit_duration, offspring_result)

            # penalty 0
            penalty_0 = self.penalty_0(s_user, self.time_user, offspring_result, DISTANCES, timeOffspring)

            # penalty i
            penalty_i = self.penalty_i(self.penalty_0_arrived_time, offspring_result, DISTANCES, timeOffspring, visit_duration)

            # total penalty
            penalty = self.penalty(self.penalty_0_result, self.penalty_i_result)

            # fitness
            fitness = self.fitness(self.offspring_total_dist, penalty)

            # evaluate
            evaluate = self.evaluate(offspring_result, fitness, pop_size)

            # best fitnes
            best_fitness.append(self.evaluate_best)
            chromosome_result = self.evaluate_selected

        # sorting best fitness
        best = np.array(sorted(best_fitness, key=lambda col: -col[len(best_fitness[0]) - 1]))
        # best fit of generations
        best = best[0, 0:len(best[0]) - 1]

        # schedule
        schedule = self.schedule(self.offspring_result, self.penalty_i_arrived, self.penalty_i_start, best)
        schedule_result = self.schedule_result
        time_window_result = []
        best_route = schedule_result[:, 0]
        print("best_route: ",best_route)
        arrvd = schedule_result[:, 1]
        finish = schedule_result[:, 2]
        for i in range(min(len(schedule_result), len(arrvd), len(finish))):
            data = {
                'Destination ' + str(i + 1) : destination[i],
                'Start at' : str(arrvd[i]),
                'Finish at' : str(finish[i])
            }
            time_window_result.append(data)
        print("time_window_result: ",time_window_result)
        # print(COORDINATE[:,0])
        # self.common_plot(0,COORDINATE[:,0],COORDINATE[:,1],False,best_route)

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

def compute_distance_cities(num_city, location):
    # build a zero array (num_city*num_city) [0,0,...]
    distance_cities = np.zeros((num_city, num_city))
    for i in range(num_city):
        for j in range(num_city):
            if i == j:
                distance_cities[i][j] = np.inf
                continue
            x = location[i]
            y = location[j]
            # calculator Euler distance
            tmp_distance = np.sqrt(sum([(float(x[0]) - float(x[1])) ** 2 for x in zip(x, y)]))
            distance_cities[i][j] = tmp_distance
    return distance_cities

def handle_data(name_file):
    global NUMBER_CITY
    global DURATION_TIME
    global DISTANCES
    global READY_TIME
    global COORDINATE

    data = read_data.read_twtsp('data/'+name_file)
    data = np.array(data)
    NUMBER_CITY = len(data)
    time = data[:,4:6]
    ready_times = []
    duration_time = []
    for i in time:
        h, m = divmod(i[0], 60)
        if h<10:
            ready_time = str.format("0%d:%02d:%02d" % (h, m, 00))
        else:
            ready_time = str.format("%d:%02d:%02d" % (h, m, 00))
        ready_times.append(ready_time)
        duration_time.append(i[1]-i[0])
    DURATION_TIME = duration_time
    # compute distances
    data = data[:, 1:3]
    COORDINATE = data

    DISTANCES = compute_distance_cities(len(data),data)
    data_time = []
    for i in range (len(ready_times)):
        d_time = ''.join(map(str, ready_times[i]))
        d_time = datetime.strptime(d_time, '%H:%M:%S').time()
        data_time.append(d_time)
    READY_TIME = data_time

def start_visit_city(name_file):
    handle_data(name_file)
    # city index start at 1
    start_city = 3
    destination = []
    visit_duration = []
    for i in range(0,NUMBER_CITY):
        if i != start_city:
            destination.append(i)
            visit_duration.append(DURATION_TIME[i])
    time_user = "00:00"
    POPULATION_SIZE = len(destination)
    mode = tsptwSolutionByGeneticAlgorithm(start_city,destination,visit_duration,time_user,
                                           pop_size=POPULATION_SIZE,crossover_rate=GA_CHOOSE_RATION,
                                           mutation_rate=MUTATE_RATIO,iteration=MAX_ITERATION)
    mode.run()

# start_city,destination,visit_duration,time_user, pop_size, crossover_rate,mutation_rate,iteration
if __name__ == '__main__':
    name_file = "TSPTW_dataset.txt"
    start_visit_city(name_file)
