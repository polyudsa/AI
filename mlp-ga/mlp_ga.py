# coding:utf-8
"""
使用遗传算法的神经网络
寻找超参数的最优解

cross_ration：交叉概率
mutate_ration：个体突变的概率
iteration：生成的循环数

individual：一组设计变量
population：当前一代的人口
offspring：下一代种群
fitness：适应度
selection：从当前一代到下一代的选择
crossover：两个个体之间基因的交叉/交叉
mutation： 突变

1. 初始设置
2. 评估
3. 选择
4. 交叉
5. 突变

"""

import random
from deap import base, creator, tools
import matplotlib.pyplot as plt
import mlp
import test


def genAlg(population=5, cross_ration=0.5, mutate_ration=0.2, iteration=5):
    """
    cross_ration: 交叉概率
    mutate_ration: 突变概率
    iteration: 迭代次数
    """
    random.seed(64)
    pop = toolbox.population(n=population)

    print("start of evolution")

    # 评估初始种群中的个体
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(" %i invalid " % len(pop))

    best_fitness = []
    # 进化开始
    for g in range(iteration):
        print(" -- %i gen --" % g)

        """ 选择 """
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        """ 交叉 """
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cross_ration:
                toolbox.mate(child1, child2)
                # 去除交叉个体的适应度
                del child1.fitness.values
                del child2.fitness.values
        
        """ 变异 """
        for mutant in offspring:
            if random.random() < mutate_ration:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        try:
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        except AssertionError:
            pass
        
        print(" %i invalid " % len(invalid_ind))

        # 下一代
        pop[:] = offspring
        # 适应度
        try:
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits)/length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
        except IndexError:
            pass
    
    print("-- iteration end -- ")

    best_ind = tools.selBest(pop, 1)[0]
    print(" best individual %s %s " % (best_ind, best_ind.fitness.values))

    return best_ind

def run_mlp(bounds):
    _mlp = mlp.MLP(dense1=bounds[0],
                    dense2=bounds[1],
                    drop1=bounds[2],
                    drop2=bounds[3],
                    batch_size=bounds[4],
                    activation=bounds[5],
                    opt=bounds[6]
                    )

    mnist_evaluation = _mlp.mlp_evaluate()
    
    return mnist_evaluation[0],
    

""" define Genetic Algorithm """

creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness= creator.FitnessMax)

# defining attributes for individual
toolbox = base.Toolbox()

# 定义生成每个参数的函数
# neuron size
toolbox.register("dense1",random.choice, (32, 64, 128, 256, 512, 1024))
toolbox.register("dense2", random.choice, (32, 64, 128, 256, 512, 1024))

# dropout late
toolbox.register("drop1", random.uniform, 0.2, 0.5)
toolbox.register("drop2", random.uniform, 0.2, 0.5)

# training
# toolbox.register("batch_size", random.choice, (16, 32, 64, 128, 256, 512))
toolbox.register("batch_size", random.choice, (16, 32, 64, 128, 256, 512))
toolbox.register("activation", random.choice, ('sigmoid','relu'))
toolbox.register("optimizer", random.choice, ('RMSprop','SGD1','SGD2','Adam'))

# register attributes to individual
toolbox.register('individual', tools.initCycle, creator.Individual,
                    (toolbox.dense1, toolbox.dense2,
                    toolbox.drop1, toolbox.drop2,
                    toolbox.batch_size, toolbox.activation, toolbox.optimizer),
                    n = 1)

# individual to population
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# 交叉定义
toolbox.register('mate', tools.cxTwoPoint)
# 变异定义
toolbox.register('mutate', tools.mutFlipBit, indpb = 0.05)
# 选择定义
toolbox.register('select', tools.selTournament, tournsize=3)

toolbox.register('evaluate', run_mlp)

best_int = genAlg(population=5, cross_ration=0.5, mutate_ration=0.2, iteration=50)


