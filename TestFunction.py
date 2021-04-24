import numpy as np

import math

def Rastrigin(chromosome):
    fitness = 3 * len(chromosome)
    for i in range(len(chromosome)):
        fitness += chromosome[i].value ** 2 - (3 * math.cos(2 * math.pi * chromosome[i].value))
    return fitness


def Schewefel(chromosome):
    alpha = 418.982887
    fitness = alpha * len(chromosome)
    for i in range(len(chromosome)):
        fitness -= chromosome[i].value * math.sin(math.sqrt(math.fabs(chromosome[i].value)))
    return fitness


def Griewank(chromosome):
    part1 = 0
    for i in range(len(chromosome)):
        part1 += chromosome[i].value ** 2
        part2 = 1
    for i in range(len(chromosome)):
        part2 *= math.cos(float(chromosome[i].value) / math.sqrt(i + 1))
    return 1 + (float(part1) / 4000.0) - float(part2)

def Ackley(chromosome):
	firstSum = 0.0
	secondSum = 0.0
	for c in chromosome:
		firstSum += c.value**2.0
		secondSum += math.cos(2.0*math.pi*c.value)
	n = float(len(chromosome))
	return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e