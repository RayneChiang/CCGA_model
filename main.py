import numpy as np
import math
import random
from TestFunction import Rastrigin
import time
class IndiString:
    def __init__(self):
        self.represent = '0b' + ''.join(map(str, np.random.randint(0, 2, 17)))
        self.value = self.__getValue__()

    def __getValue__(self):
        value = int(self.represent[3:20], 2) / math.pow(2, decimal_point)
        return value if self.represent[2] == '1' else - value

    def __setRepresent__(self, string):
        self.represent = string


def init_pop(population_size, min_value, max_value, n):
    pop_list = []
    while len(pop_list) < population_size:
        pop_individual = []
        while len(pop_individual) < n:
            new_indi = IndiString()
            indi_value = new_indi.__getValue__()
            if min_value <= indi_value <= max_value:
                pop_individual.append(new_indi)
        pop_list.append(pop_individual)
    return pop_list


def crossover(func, init_list, crossover_rate):
    individual_A = random.choice(init_list)
    individual_B = random.choice(init_list)
    individual_C = random.choice(init_list)
    individual_D = random.choice(init_list)
    parent_A = individual_A if func(individual_A) < func(individual_B) else individual_B
    parent_B = individual_C if func(individual_C) < func(individual_D) else individual_D
    if random.uniform(0, 1) < crossover_rate:
        point_one = random.randint(2, 20)
        point_two = random.randint(point_one, 20)
        child_list = []
        if random.uniform(0, 1) < crossover_rate:
            for i in range(len(parent_A)):
                new_string = parent_A[i].represent[0:point_one] + parent_B[i].represent[point_one:point_two] + parent_A[
                                                                                                                   i].represent[
                                                                                                               point_two:20]
                parent_A[0].__setRepresent__(new_string)
                child_list.append(parent_A[0])
            return child_list
        else:
            return parent_A[0:point_one] + parent_B[point_one:point_two] + parent_A[point_two:20]
    else:
        return parent_A if func(parent_A) < func(parent_B) else parent_B


def crossover_ccga(func, init_list, crossover_rate):
    individual_A = random.choice(init_list)
    individual_B = random.choice(init_list)
    individual_C = random.choice(init_list)
    individual_D = random.choice(init_list)
    parent_A = individual_A if func(individual_A) < func(individual_B) else individual_B
    parent_B = individual_C if func(individual_C) < func(individual_D) else individual_D
    if random.uniform(0, 1) < crossover_rate:
        point_one = random.randint(2, 20)
        point_two = random.randint(point_one, 20)
        # new_indi = IndiString()
        new_string = parent_A[0].represent[0:point_one] + parent_B[0].represent[point_one:point_two] \
                     + parent_A[0].represent[point_two:20]
        parent_A[0].__setRepresent__(new_string)
        return parent_A
    else:
        return parent_A if func(parent_A) < func(parent_B) else parent_B


def mutate(child_list):
    for indi in child_list:
        origin_string = indi.represent
        set_string = '0b'
        for char in origin_string[2:20]:
            if random.uniform(0, 1) < 1 / 16:
                mute = '0' if char == '1' else '1'
                set_string = set_string + mute
            else:
                set_string = set_string + char
            indi.__setRepresent__(set_string)
    return child_list


def ccga_model(pop_list, func, epoch):
    eval_list = list(map(func, pop_list))
    fitness_max = [min(eval_list)]
    fitness_min = [max(eval_list)]
    for i in range(epoch):
        # applying scaling window
        species_list = []
        gen_list = [pop_list[np.nanargmin(eval_list)]]
        # applying scaling window
        # fmin = fitness_min[0] if i < 6 else fitness_min[i - 5]
        # fitness_list = fmin - eval_list
        # # fitness_list = np.int64(fitness_list >= 0)
        # # apply elitist strategy
        # if (sum(fitness_list) != 0):
        #     proportion = fitness_list / sum(fitness_list) * len(fitness_list)
        #     proportion_list = np.array(list(map(round, proportion)))
        # else:
        #     proportion_list = np.ones(len(fitness_list))
        # # pure strategy select
        # select_1 = np.where(proportion_list == 1)[0]
        # select_2 = np.where(proportion_list > 1)[0]
        # if len(select_2) != 0:
        #     pop_list = [pop_list[i] for i in select_1] + 2 * [pop_list[i] for i in select_2]
        # else:
        #     pop_list = [pop_list[i] for i in select_1]
        for s in range(0, num_parameter):
            s_list = []
            s_return_list = []
            for pop in pop_list:
                s_list.append([pop[s]])
            s_eval_list = list(map(func, s_list))
            fmin = max(s_eval_list)
            f_list = fmin - s_eval_list
            if(sum(f_list)!=0):
                proportion = f_list / sum(f_list) * len(s_list)
                proportion_list = np.array(list(map(round, proportion)))
            else:
                proportion_list = np.ones(len(f_list))
            select_1 = np.where(proportion_list == 1)[0]
            select_2 = np.where(proportion_list > 1)[0]
            if len(select_2) != 0:
                s_list = [s_list[i] for i in select_1] + 2 * [s_list[i] for i in select_2]
            else:
                s_list = [s_list[i] for i in select_1]
            while len(s_return_list) < len(init_list) - 1:
                # two-point crossover
                child = crossover_ccga(func, s_list, crossover_rate=0.6)
                if random.uniform(0, 1) < 1 / 16:
                    child = mutate(child)
                s_return_list.extend(child)
            species_list.append(s_return_list)
        # re-evaluate
        gen_list.extend(list(map(list, zip(*species_list))))
        eval_list = list(map(func, gen_list))
        fitness_max.append(min(eval_list))
        fitness_min.append(max(eval_list))
        pop_list = gen_list

    return pop_list, fitness_max


def ga_model(pop_list, func, epoch):
    eval_list = list(map(func, pop_list))
    fitness_max = [min(eval_list)]
    fitness_min = [max(eval_list)]
    for i in range(epoch):
        # applying scaling window
        fmin = fitness_min[0] if i < 6 else fitness_min[i - 5]
        fitness_list = fmin - eval_list
        # fitness_list = np.int64(fitness_list >= 0)
        # apply elitist strategy
        gen_list = [pop_list[np.nanargmax(fitness_list)]]
        if (sum(fitness_list) != 0):
            proportion = fitness_list / sum(fitness_list) * len(fitness_list)
            proportion_list = np.array(list(map(round, proportion)))
        else:
            proportion_list = np.ones(len(fitness_list))
        # pure strategy select
        select_1 = np.where(proportion_list == 1)[0]
        select_2 = np.where(proportion_list > 1)[0]
        if len(select_2) != 0:
            pop_list = [pop_list[i] for i in select_1] + 2 * [pop_list[i] for i in select_2]
        else:
            pop_list = [pop_list[i] for i in select_1]
        while len(gen_list) < len(init_list):
            # two-point crossover
            child = crossover(func, pop_list, crossover_rate=0.6)
            if random.uniform(0, 1) < 1 / 16:
                child = mutate(child)
            gen_list.append(child)
        # re-evaluate
        eval_list = list(map(func, gen_list))
        fitness_max.append(min(eval_list))
        fitness_min.append(max(eval_list))
        pop_list = gen_list
    return pop_list, fitness_max


if __name__ == '__main__':
    decimal_point = 13
    pupulation_size = 100
    num_parameter = 20
    init_list = init_pop(population_size=100, min_value=-5.12, max_value=5.12, n=num_parameter)
    print(time.ctime())
    pop_list_ga, fitness_ga = ga_model(init_list, Rastrigin, 100)
    print(fitness_ga)
    np.save('Rastrigin_ga_result.npy', np.array(fitness_ga))
    print(time.ctime())
    pop_list_ccga, fitness_ccga = ccga_model(init_list, Rastrigin, 100)
    print(fitness_ccga)
    np.save('Rastrigin_ccga_result.npy', np.array(fitness_ccga))
    print(time.ctime())



