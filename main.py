import numpy as np
import math
import random
from TestFunction import Rastrigin, Schewefel, Griewank, Ackley, Xin_She_Yang, Xin_She_Yang_2
import time
from tqdm import tqdm
import pandas as pd
from decimal import Decimal
from PyBenchFCN import SingleObjectiveProblem as SOP


class IndiString:
    def __init__(self, string, divide):
        self.represent = string
        self.divide_point = divide
        self.value = self.__setValue__()

    def __setValue__(self):
        num = int(self.represent[1:17], 2) / math.pow(2, self.divide_point)
        if self.represent[0] == 0:
            num = - num
        while num > max_value or num < min_value:
            num = num / 2
            self.divide_point = self.divide_point + 1
        return num

    def __getValue__(self):
        num = int(self.represent[1:17], 2) / math.pow(2, self.divide_point)
        if self.represent[0] == 0:
            num = - num
        return num


def init_pop(population_size, n):
    pop_list = []
    while len(pop_list) < population_size:
        pop_individual = []
        while len(pop_individual) < n:
            new_indi = IndiString(initStringValue(), 0)
            pop_individual.append(new_indi)
        pop_list.append(pop_individual)
    return pop_list


def initStringValue():
    seed = "01"
    sa = []
    for i in range(17):
        sa.append(seed[np.random.randint(0, 2)])
    salt = ''.join(sa)
    return salt


def get_random(key_list):
    L = len(key_list)
    i = np.random.randint(0, L)
    return key_list[i]


def crossover_mute(func, list, crossover_rate):
    individual_A = get_random(list)
    individual_B = get_random(list)
    individual_C = get_random(list)
    individual_D = get_random(list)
    parent_A = individual_A if func(individual_A) < func(individual_B) else individual_B
    parent_B = individual_C if func(individual_C) < func(individual_D) else individual_D

    child_list = []
    if random.uniform(0, 1) < crossover_rate:
        for i in range(len(parent_A)):
            str_len = min((len(parent_A[i].represent), len(parent_B[i].represent)))
            point_one = random.randint(0, str_len)
            point_two = random.randint(point_one, str_len)
            new_string = parent_A[i].represent[0:point_one] + parent_B[i].represent[point_one:point_two] + \
                         parent_A[i].represent[point_two:]
            new_string = mutate(new_string)
            new_indi = IndiString(new_string, parent_A[i].divide_point)
            child_list.append(new_indi)
        return child_list
    else:
        child = parent_A if func(parent_A) < func(parent_B) else parent_B
        return child


def crossover_mute_ccga(func, list, crossover_rate):
    individual_A = get_random(list)
    individual_B = get_random(list)
    individual_C = get_random(list)
    individual_D = get_random(list)
    parent_A = individual_A if func(individual_A) < func(individual_B) else individual_B
    parent_B = individual_C if func(individual_C) < func(individual_D) else individual_D
    string_len = len(parent_A[0].represent)
    if random.uniform(0, 1) < crossover_rate:
        point_one = random.randint(0, string_len)
        point_two = random.randint(point_one, string_len)
        new_string = parent_A[0].represent[0:point_one] + parent_B[0].represent[point_one:point_two] \
                     + parent_A[0].represent[point_two:]
        new_string = mutate(new_string)
        new_indi = IndiString(new_string, parent_A[0].divide_point)
        return [new_indi]
    else:
        child = parent_A if func(parent_A) < func(parent_B) else parent_B
        return child


def mutate(string):
    origin_string = string
    set_string = ''
    for char in origin_string[0:]:
        if random.uniform(0, 1) < 1 / 16:
            mute = '0' if char == '1' else '1'
            set_string = set_string + mute
        else:
            set_string = set_string + char
    return set_string


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


# extend s*d
def ccga_svg_model(pop_list, func, epoch):
    eval_list = list(map(func, pop_list))
    fitness_max = [np.min(eval_list)]
    fitness_min = [np.max(eval_list)]
    for _ in tqdm(range(epoch)):
        species_list = []
        gen_list = [pop_list[np.nanargmin(eval_list)]]
        for s in range(0, 10):
            s_list = []
            for pop in pop_list:
                s_list.append([pop[s*10], pop[s*10+1], pop[s*10+2], pop[s*10+3], pop[s*10+4],
                               pop[s*10+5], pop[s*10+6], pop[s*10+7], pop[s*10+8], pop[s*10+9]])
            s_eval_list = list(map(func, s_list))
            fmin = np.max(s_eval_list)
            f_list = fmin - s_eval_list
            s_list = scaling_window(f_list, s_list)
            a_list = []
            for i in range(0, 100):
                s_return_list = []
                while len(s_return_list) < len(init_list):
                    child = crossover_mute(func, s_list, crossover_rate=0.6)
                    s_return_list.extend(child)
                a_list.append(s_return_list)
            species_list.extend(a_list)
        # re-evaluate
        gen_list.extend(list(map(list, zip(*species_list)))[0:-1])
        eval_list = list(map(func, gen_list))
        fitness_max.append(np.min(eval_list))
        fitness_min.append(np.max(eval_list))
        pop_list = gen_list

    return pop_list, fitness_max



def ccga_svgr_model(pop_list, func, epoch):
    eval_list = list(map(func, pop_list))
    fitness_max = [np.min(eval_list)]
    fitness_min = [np.max(eval_list)]
    for _ in tqdm(range(epoch)):
        species_list = []
        gen_list = [pop_list[np.nanargmin(eval_list)]]
        for s in range(0, 5):
            s_list = []
            for pop in pop_list:
                s_list.append([pop[random.randint(0, num_parameter-1)], pop[random.randint(0, num_parameter-1)],pop[random.randint(0, num_parameter-1)],
                               pop[random.randint(0, num_parameter-1)],pop[random.randint(0, num_parameter-1)]])
            s_eval_list = list(map(func, s_list))
            fmin = np.max(s_eval_list)
            f_list = fmin - s_eval_list
            s_list = scaling_window(f_list, s_list)
            a_list = []
            for i in range(0, 200):
                s_return_list = []
                while len(s_return_list) < len(init_list):
                    child = crossover_mute(func, s_list, crossover_rate=0.6)
                    s_return_list.extend(child)
                a_list.append(s_return_list)
            species_list.extend(a_list)
        # re-evaluate
        gen_list.extend(list(map(list, zip(*species_list)))[0:-1])
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
        gen_list = [pop_list[np.nanargmin(eval_list)]]
        fmin = fitness_min[0] if i < 6 else fitness_min[i - 5]
        fitness_list = fmin - eval_list
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
    pupulation_size = 100
    num_parameter = 20
    min_value = -5.12
    max_value = 5.12
    init_list = init_pop(population_size=100, n=num_parameter)
    pop_list_ccga, fitness_ccga = ccga_model(init_list, Rastrigin, 1000)
    np.save('Rccga.npy', np.array(fitness_ccga))
    pop_list_svg_ccga, fitness_svg_ccga = ccga_svgr_model(init_list, Rastrigin, 1000)
    np.save('R_s_ccga.npy', np.array(fitness_svg_ccga))

    pupulation_size = 100
    num_parameter = 10
    min_value = -500
    max_value = 500
    init_list = init_pop(population_size=100, n=num_parameter)
    pop_list_ga, fitness_max= ccga_model(init_list, Schewefel, 1000)
    np.save('S_ga.npy', np.array(fitness_max))
    pop_list_ccga, fitness_ccga = ccga_svgr_model(init_list, Schewefel, 1000)
    np.save('S_s_ccga.npy', np.array(fitness_ccga))




    pupulation_size = 100
    num_parameter = 30
    min_value = -30
    max_value = 30
    init_list = init_pop(population_size=100, n=num_parameter)
    pop_list_ga, fitness_max = ccga_model(init_list, Ackley, 1000)
    np.save('A_ga.npy', np.array(fitness_max))
    pop_list_ccga, fitness_ccga = ccga_svgr_model(init_list, Ackley, 1000)
    np.save('A_s_ccga.npy', np.array(fitness_ccga))

    # #
    pupulation_size = 100
    num_parameter = 10
    min_value = -512
    max_value = 512
    init_list = init_pop(population_size=100, n=num_parameter)
    pop_list_ga, fitness_max= ccga_model(init_list, Griewank, 1000)
    np.save('G_ga.npy', np.array(fitness_max))
    pop_list_ccga, fitness_ccga = ccga_svgr_model(init_list, Griewank, 1000)
    np.save('G_svg_ccga.npy', np.array(fitness_ccga))
    #
    #


    pupulation_size = 100
    num_parameter = 1000
    min_value = -2*np.pi
    max_value = 2*np.pi
    init_list = init_pop(population_size=100, n=num_parameter)
    # # pop_list_ga, fitness_max= ga_model(init_list, Xin_She_Yang_2, 100)
    # # print(fitness_max)
    # # np.save('X_ga_2_400.npy', np.array(fitness_max))
    pop_list_ccga, fitness_ccga = ccga_model(init_list, Xin_She_Yang, 10)
    print(fitness_ccga)
    np.save('X_ccga_1_1000.npy', np.array(fitness_ccga))
    pop_list_ccga, fitness_ccga = ccga_svgr_model(init_list, Xin_She_Yang, 10)
    print(fitness_ccga)
    np.save('X_svg_ccga_1_1000.npy', np.array(fitness_ccga))
