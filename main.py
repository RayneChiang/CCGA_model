import numpy as np
import math
import random
from TestFunction import Rastrigin, Schewefel, Griewank, Ackley
import time
from tqdm import tqdm
import pandas as pd
from decimal import Decimal


class IndiString:
    def __init__(self, min_v, max_v):
        self.value = round(random.uniform(min_v, max_v), 10)
        self.represent = self.__initStringValue__()

    def __setRepresent__(self, string):
        self.represent = string

        self.value = int(self.represent, 2) / math.pow(10, decimal_point)
        self.__checkValue__()

    def __checkValue__(self):
        while self.value > max_value or self.value < min_value:
                self.value = self.value/5

    def __initStringValue__(self):
        number = int(self.value * math.pow(10, decimal_point))
        while number > 65536 or number < -65536:
            number = int(number/10)
        return bin(number)


def init_pop(population_size, min_v, max_v, n):
    pop_list = []
    while len(pop_list) < population_size:
        pop_individual = []
        while len(pop_individual) < n:
            new_indi = IndiString(min_v=min_v, max_v=max_v)
            pop_individual.append(new_indi)
        pop_list.append(pop_individual)
    return pop_list


def get_random(key_list):
    L = len(key_list)
    i = np.random.randint(0, L)
    return key_list[i]


def crossover_mute(func, init_list, crossover_rate):
    individual_A = get_random(init_list)
    individual_B = get_random(init_list)
    individual_C = get_random(init_list)
    individual_D = get_random(init_list)
    parent_A = individual_A if func(individual_A) < func(individual_B) else individual_B
    parent_B = individual_C if func(individual_C) < func(individual_D) else individual_D
    if random.uniform(0, 1) < crossover_rate:
        point_one = random.randint(0, num_parameter+1)
        point_two = random.randint(point_one, num_parameter+1)
        child_list = []
        if random.uniform(0, 1) < crossover_rate:
            for i in range(len(parent_A)):
                str_len = min((len(parent_A[i].represent), len(parent_B[i].represent)))
                new_indi = IndiString(min_v=min_value, max_v=max_value)
                point_one = random.randint(3, str_len)
                point_two = random.randint(point_one, str_len)
                new_string = parent_A[i].represent[0:point_one] + parent_B[i].represent[point_one:point_two] + \
                             parent_A[i].represent[point_two:]
                if len(new_string) > 16:
                    new_indi.__setRepresent__(new_string)
                    mutate(new_indi)
                    child_list.append(new_indi)
                else:
                    child_list.append(parent_A[i])
            return child_list
        else:
            return parent_A[0:point_one] + parent_B[point_one:point_two] + parent_A[point_two:]
    else:
        return parent_A if func(parent_A) < func(parent_B) else parent_B


def crossover_mute_ccga(func, init_list, crossover_rate):
    individual_A = get_random(init_list)
    individual_B = get_random(init_list)
    individual_C = get_random(init_list)
    individual_D = get_random(init_list)
    parent_A = individual_A if func(individual_A) < func(individual_B) else individual_B
    parent_B = individual_C if func(individual_C) < func(individual_D) else individual_D
    string_len = len(parent_A[0].represent)
    if random.uniform(0, 1) < crossover_rate and string_len>16:
        point_one = random.randint(3, string_len)
        point_two = random.randint(point_one, string_len)
        new_indi = IndiString(min_v=min_value, max_v=max_value)
        new_string = parent_A[0].represent[0:point_one] + parent_B[0].represent[point_one:point_two] \
                     + parent_A[0].represent[point_two:]
        new_indi.__setRepresent__(new_string)
        mutate(new_indi)
        return [new_indi]
    else:
        return (parent_A) if func(parent_A) < func(parent_B) else parent_B


def mutate(individual):
    if random.uniform(0, 1) < 1 / 16:
        origin_string = individual.represent
        set_string = '0b'
        for char in origin_string[3:-1]:
            if random.uniform(0, 1) < 1 / 16:
                mute = '0' if char == '1' else '1'
                set_string = set_string + mute
            else:
                set_string = set_string + char
            if len(set_string) > 16:
                individual.__setRepresent__(set_string)


def ccga_model(pop_list, func, epoch):
    eval_list = list(map(func, pop_list))
    fitness_max = [np.min(eval_list)]
    fitness_min = [np.max(eval_list)]
    for _ in tqdm(range(epoch)):
        species_list = []
        gen_list = [pop_list[np.nanargmin(eval_list)]]
        for s in range(0, num_parameter):
            s_list = []
            s_return_list = []
            for pop in pop_list:
                s_list.append([pop[s]])
            s_eval_list = list(map(func, s_list))
            fmin = np.max(s_eval_list)
            f_list = fmin - s_eval_list
            s_list = scaling_window(f_list, s_list)
            while len(s_return_list) < len(init_list) - 1:
                child = crossover_mute_ccga(func, s_list, crossover_rate=0.6)
                s_return_list.extend(child)
            species_list.append(s_return_list)
        # re-evaluate
        gen_list.extend(list(map(list, zip(*species_list))))
        eval_list = list(map(func, gen_list))
        fitness_max.append(np.min(eval_list))
        fitness_min.append(np.max(eval_list))
        pop_list = gen_list

    return pop_list, fitness_max


def scaling_window(fitness_list, pop_list):
    if sum(fitness_list) != 0:
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
    return pop_list


def ga_model(pop_list, func, epoch):
    eval_list = list(map(func, pop_list))
    fitness_max = [np.min(eval_list)]
    fitness_min = [np.max(eval_list)]
    for i in tqdm(range(epoch)):
        # applying scaling window
        fmin = fitness_min[0] if i < 6 else fitness_min[i - 5]
        fitness_list = fmin - eval_list
        # fitness_list = np.int64(fitness_list >= 0)
        # apply elitist strategy
        gen_list = [pop_list[np.nanargmax(fitness_list)]]
        # scaling window
        pop_list = scaling_window(fitness_list, pop_list)
        while len(gen_list) < len(init_list):
            # two-point crossover
            child = crossover_mute(func, pop_list, crossover_rate=0.6)
            gen_list.append(child)
        # re-evaluate
        eval_list = list(map(func, gen_list))
        fitness_max.append(np.min(eval_list))
        fitness_min.append(np.max(eval_list))
        pop_list = gen_list
    return pop_list, fitness_max


if __name__ == '__main__':
    decimal_point = 4
    pupulation_size = 100
    num_parameter = 20
    min_value = -5.12
    max_value = 5.12
    init_list = init_pop(population_size=100, min_v=min_value, max_v=max_value, n=num_parameter)
    pop_list_ga, fitness_ga = ga_model(init_list, Rastrigin, 100000)
    np.save('Rastrigin_ga.npy', np.array(fitness_ga))
    pop_list_ccga, fitness_ccga = ccga_model(init_list, Rastrigin, 100000)
    np.save('Rastrigin_ccga.npy', np.array(fitness_ccga))


    decimal_point = 2
    pupulation_size = 100
    num_parameter = 10
    min_value = -500
    max_value = 500
    init_list = init_pop(population_size=100, min_v=min_value, max_v=max_value, n=num_parameter)
    pop_list_ga, fitness_ga = ga_model(init_list, Schewefel, 100000)
    np.save('Schewefel_ga.npy', np.array(fitness_ga))
    pop_list_ccga, fitness_ccga = ccga_model(init_list, Schewefel, 100000)
    np.save('Schewefel_ccga.npy', np.array(fitness_ccga))


    decimal_point = 1
    pupulation_size = 100
    num_parameter = 10
    min_value = -512
    max_value = 512
    init_list = init_pop(population_size=100, min_v=min_value, max_v=max_value, n=num_parameter)
    pop_list_ga, fitness_ga = ga_model(init_list, Griewank, 100000)
    np.save('Griewank_ga.npy', np.array(fitness_ga))
    pop_list_ccga, fitness_ccga = ccga_model(init_list, Griewank, 100000)
    np.save('Griewank_ccga.npy', np.array(fitness_ccga))


    decimal_point = 5
    pupulation_size = 100
    num_parameter = 30
    min_value = -30
    max_value = 30
    init_list = init_pop(population_size=100, min_v=min_value, max_v=max_value, n=num_parameter)
    pop_list_ga, fitness_ga = ga_model(init_list, Ackley, 10000)
    np.save('Ackley_ga.npy', np.array(fitness_ga))
    pop_list_ccga, fitness_ccga = ccga_model(init_list, Ackley, 100000)
    np.save('Ackley_ccga.npy', np.array(fitness_ccga))