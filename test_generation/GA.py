import sys
sys.path.append('.')
import numpy as np
import random
import time
from img_mutators import Mutators
# from DiffTesting2 import create_image_indvs


def create_image_indvs(img, num):
    indivs = []
    indivs.append(img)
    for i in range(num-1):
        indivs.append(Mutators.mutate(img, img))
    return np.array(indivs)


class Population():
    def __init__(self,
                 individuals,
                 mutation_function,
                 fitness_compute_function,
                 save_function,
                 ground_truth,
                 seed,
                 max_iteration,
                 tour_size=20, cross_rate=0.5, mutate_rate=0.005, max_time=30):
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.individuals = individuals # a list of individuals, current is numpy
        self.ground_truth = ground_truth
        self.tournament_size = tour_size
        self.fitness = None   # a list of fitness values
        self.pop_size = len(self.individuals)
        self.mutation_func = mutation_function
        self.fitness_fuc = fitness_compute_function
        self.save_function = save_function
        self.order = []
        self.best_fitness = -1000
        self.best_top2 = 1
        self.best_top3 = 1
        self.best_var = 1
        self.success = 0
        self.check = 0
        # for i in range(max_trials):
        start_time = time.time()
        self.i = 0
        self.fitness_change = 0

        while True:
            # (select_ind, select_prob, select_score, select_correct), probfitness, varscore, classresults = self.evolvePopulation()
            select_ind, classresults = self.evolvePopulation()
            if self.i >= max_iteration:

                self.save_function(self.individuals, self.i)
                break
            print(len(select_ind))
            if len(select_ind) > 0:
                self.save_function(select_ind, self.i)
                break

            if time.time() - start_time > max_time:
                break
            self.i += 1

            total_time = time.time() - start_time
            speed = total_time/self.i

            print(" Total generation: %d, best fitness:%.9f, top2: %.9f, top3: %.9f, speed: %2f" % (self.i, self.best_fitness, self.best_top2, self.best_top3, speed))

    def crossover(self, ind1, ind2):
        mask = np.random.rand(*ind1.shape) < self.cross_rate
        return mask * ind1 + (1 - mask) * ind2

    def evolvePopulation(self):

        # fitness1, fitness2, varscore, start_change, probfitness, outputs, classresults = self.fitness_fuc(self.individuals)
        fitness1, fitness2, outputs, probfitness, probfitness2, classresults = self.fitness_fuc(
            self.individuals, self.ground_truth)

        """
            sorted_fitness_indexes: the ordered indexes based on fitness value
            sorted_fitness_indexes[0] is the index of individual with the best fitness
        """

        """
            tournaments: randomly select a tournament from the individuals and get the indv with best fittness
            Instead of select from individuals , we select from the sorted indexes (i.e., sorted_fitness_indexes) randomly.
            sorted_fitness_indexes[order_seq1[0]] is the index of indivitual with best fitness in the selected tournament.
        """
        sorted_fitness1_indexes = sorted(range(len(fitness1)), key=lambda k: fitness1[k], reverse=True)
        sorted_fitness2_indexes = sorted(range(len(fitness2)), key=lambda k: fitness2[k], reverse=True)
        new_indvs = []
        best_index1 = sorted_fitness1_indexes[0]
        best_index2 = sorted_fitness2_indexes[1]
        for i, item in enumerate(self.individuals):
            if i == best_index1 or i == best_index2:  # keep best
                new_indvs.append(item)
            else:
                order_seq1 = np.sort(np.random.choice(np.arange(self.pop_size), self.tournament_size, replace=False))
                order_seq2 = np.sort(np.random.choice(np.arange(self.pop_size), self.tournament_size, replace=False))

                # Finally, we get two best individual in the two tournaments.
                first_individual = self.individuals[sorted_fitness1_indexes[order_seq1[0]]]
                second_individual = self.individuals[
                    sorted_fitness2_indexes[order_seq2[0] if order_seq2[0] != order_seq1[0] else order_seq2[1]]]
                # Cross over
                ind = self.crossover(first_individual, second_individual)
                if random.uniform(0, 1) < self.mutate_rate:
                    ind = self.mutation_func(ind)
                new_indvs.append(ind)
        self.individuals = np.array(new_indvs)
        self.best_top2 = probfitness[best_index1]
        self.best_top3 = probfitness2[best_index2]
        self.best_fitness = fitness2[best_index1]
        return outputs, classresults


class Population2():
    def __init__(self,
                 individuals,
                 mutation_function,
                 fitness_compute_function,
                 save_function,
                 ground_truth,
                 seed,
                 max_iteration,
                 tour_size=20, cross_rate=0.5, mutate_rate=0.005, max_time=30):
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.individuals = individuals # a list of individuals, current is numpy
        self.ground_truth = ground_truth
        self.tournament_size = tour_size
        self.fitness = None   # a list of fitness values
        self.pop_size = len(self.individuals)
        self.mutation_func = mutation_function
        self.fitness_fuc = fitness_compute_function
        self.save_function = save_function
        self.order = []
        self.best_fitness = -1000
        self.best_top2 = 1
        self.best_top3 = 1
        self.best_var = 1
        self.success = 0
        self.check = 0
        # for i in range(max_trials):
        start_time = time.time()
        self.i = 0
        self.fitness_change = 0

        while True:
            # (select_ind, select_prob, select_score, select_correct), probfitness, varscore, classresults = self.evolvePopulation()
            select_ind, classresults = self.evolvePopulation()
            if self.i >= max_iteration:

                self.save_function(self.individuals, self.i)
                break
            print(len(select_ind))
            if len(select_ind) > 0:
                self.save_function(select_ind, self.i)
                break

            if time.time() - start_time > max_time:
                break
            self.i += 1

            total_time = time.time() - start_time
            speed = total_time/self.i

            print(" Total generation: %d, best fitness:%.9f, top2: %.9f, top3: %.9f, speed: %2f" % (self.i, self.best_fitness, self.best_top2, self.best_top3, speed))

    def crossover(self, ind1, ind2):
        mask = np.random.rand(*ind1.shape) < self.cross_rate
        return mask * ind1 + (1 - mask) * ind2

    def evolvePopulation(self):

        # fitness1, fitness2, varscore, start_change, probfitness, outputs, classresults = self.fitness_fuc(self.individuals)
        fitness1, fitness2, outputs, probfitness, probfitness2, classresults = self.fitness_fuc(
            self.individuals, self.ground_truth)

        """
            sorted_fitness_indexes: the ordered indexes based on fitness value
            sorted_fitness_indexes[0] is the index of individual with the best fitness
        """

        """
            tournaments: randomly select a tournament from the individuals and get the indv with best fittness
            Instead of select from individuals , we select from the sorted indexes (i.e., sorted_fitness_indexes) randomly.
            sorted_fitness_indexes[order_seq1[0]] is the index of indivitual with best fitness in the selected tournament.
        """
        sorted_fitness1_indexes = sorted(range(len(fitness1)), key=lambda k: fitness1[k], reverse=True)
        new_indvs = []
        best_index1 = sorted_fitness1_indexes[0]
        # best_index2 = sorted_fitness2_indexes[1]
        for i, item in enumerate(self.individuals):
            if i == best_index1:  # keep best
                new_indvs.append(item)
            else:
                order_seq1 = np.sort(np.random.choice(np.arange(self.pop_size), self.tournament_size, replace=False))
                # Finally, we get two best individual in the two tournaments.
                first_individual = self.individuals[sorted_fitness1_indexes[order_seq1[0]]]
                second_individual = self.individuals[sorted_fitness1_indexes[order_seq1[2]]]
                # Cross over
                ind = self.crossover(first_individual, second_individual)
                if random.uniform(0, 1) < self.mutate_rate:
                    ind = self.mutation_func(ind)
                new_indvs.append(ind)
        self.individuals = np.array(new_indvs)
        self.best_top2 = probfitness[best_index1]
        self.best_top3 = probfitness[best_index1]
        self.best_fitness = fitness1[best_index1]
        return outputs, classresults
