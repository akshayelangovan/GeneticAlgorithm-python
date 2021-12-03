from random import randint
from random import random
import numpy as np

class Population:
    def __init__(self, no_of_chromosomes, length_of_chromosome, problem):
        self.Chromosome = []
        self.Fitness = []
        for i in range(no_of_chromosomes):
            gene = []
            for j in range(length_of_chromosome):
                gene.append(randint(problem.lb, problem.ub))
            self.Chromosome.append(gene)
            self.Fitness.append(problem.obj(gene))

    def selection(self):
        if any(x<0 for x in self.Fitness):
            scaled_fitness = [x + abs(min(self.Fitness)) for x in self.Fitness]
            normalized_fitness = [x/sum(scaled_fitness) for x in scaled_fitness]
        else:
            normalized_fitness = [x/sum(self.Fitness) for x in self.Fitness]

        normalized_fitness = np.array(normalized_fitness)
        sorted_idx = np.argsort(-normalized_fitness)
        temp_chrom_list = []
        temp_fitness_list = []
        temp_nfitness_list = []
        for i in sorted_idx:
            temp_chrom_list.append(self.Chromosome[i])
            temp_fitness_list.append(self.Fitness[i])
            temp_nfitness_list.append(normalized_fitness[i])

        cumsum = [0] * len(self.Fitness)
        for i in range(len(self.Fitness)):
            for j in range(len(self.Fitness)):
                cumsum[i] = cumsum[i] + temp_nfitness_list[j]

        R = random()
        parent1_idx = len(self.Fitness)-1
        for i in range(len(cumsum)):
            if R > cumsum[i]:
                parent1_idx = i-1
                break

        parent2_idx = parent1_idx
        while_loop_stop = 0
        while parent2_idx==parent1_idx:
            while_loop_stop = while_loop_stop + 1
            R = random()
            if while_loop_stop>20:
                break
            for i in range(len(cumsum)):
                if R>cumsum[i]:
                    parent2_idx = i-1;
                    break
        parent1 = temp_chrom_list[parent1_idx]
        parent2 = temp_chrom_list[parent2_idx]
        return parent1,parent2

    def crossover(self,parent1, parent2, Pc, crossovercase):
        if crossovercase == 1:
            cross_p = randint(0,len(parent1)-1)
            child1 = parent1[0:cross_p] + parent2[cross_p:]
            child2 = parent2[0:cross_p] + parent1[cross_p:]
        elif crossovercase == 2:
            cross_p1 = randint(0,len(parent1)-1)
            cross_p2 = cross_p1
            while cross_p2 == cross_p1:
                cross_p2 = randint(0, len(parent1) - 1)

            if  cross_p1>cross_p2:
                varhold = cross_p1
                cross_p1 = cross_p2
                cross_p2 = varhold

            child1 = parent1[0:cross_p1] + parent2[cross_p1:cross_p2] + parent1[cross_p2:]
            child2 = parent2[0:cross_p1] + parent1[cross_p1:cross_p2] + parent2[cross_p2:]
        else:
            raise ValueError('Error: crossovercase is only defined for 1 and 2.')
        if random()>Pc:
            child1 = parent1
        if random()>Pc:
            child2 = parent2
        return child1, child2

    def mutation(self, child, Pm, lb, ub):
        for i in range(len(child)):
            R = random()
            if R<Pm:
                child[i] = randint(lb,ub)
        return child

    def inversion(self, child, Pi):
        if random() < Pi:
            cross_p1 = randint(1, len(child) - 2)
            cross_p2 = cross_p1
            while cross_p2 == cross_p1:
                cross_p2 = randint(1, len(child) - 2)
            if cross_p1 > cross_p2:
                varhold = cross_p1
                cross_p1 = cross_p2
                cross_p2 = varhold

            invert = child[cross_p1:cross_p2]
            invert.reverse()
            child = child[:cross_p1] + invert + child[cross_p2:]
            return child
        else:
            return child

    def elitism(self, Pop, newPop, Er):
        M = len(Pop.Chromosome)
        #print(M)
        Elite_no = round(M*Er)
        #print(Elite_no)
        #print(Pop.Fitness)
        fitness_array = np.array(Pop.Fitness)
        #print(Pop.Chromosome)
        idx = np.argsort(-fitness_array)
        for k in range(Elite_no):
            newPop.Chromosome[k] = Pop.Chromosome[idx[k]]
            newPop.Fitness[k] = Pop.Fitness[idx[k]]
        return newPop






def GA(M,N,MaxGen,Pc,Pm,Pi,Er,Problem):
    cgcurve = [0] * MaxGen
    pop = Population(M,N,Problem)
    print(pop.Chromosome)
    cgcurve[0] = max(pop.Fitness)

    for g in range(MaxGen):
        print('Generation #',g+1)
        for i in range(M):
            pop.Fitness[i] = Problem.obj(pop.Chromosome[i])

        newPop = pop
        for k in range (0,M,2):
            #Selection of parents
            parent1, parent2 = pop.selection()

            #Crossover operation
            child1, child2 = pop.crossover(parent1, parent2, Pc, Problem.crossovercase)

            #Mutation operation
            child1 = pop.mutation(child1, Pm, Problem.lb, Problem.ub)
            child2 = pop.mutation(child2, Pm, Problem.lb, Problem.ub)

            #Inversion operation
            child1 = pop.inversion(child1,Pi)
            child2 = pop.inversion(child2, Pi)

            newPop.Chromosome[k] = child1
            newPop.Chromosome[k+1] = child2

        for i in range(M):
            #print(newPop.Chromosome[i])
            newPop.Fitness[i] = Problem.obj(newPop.Chromosome[i])

        #Elitism
        newPop = pop.elitism(pop, newPop, Er)
        pop = newPop

        cgcurve[g] = max(pop.Fitness)
    return pop,cgcurve

class Problem:
    def __init__(self,lb,ub):
        self.lb = lb
        self.ub = ub
        self.crossovercase = 1

    def obj(self, n):
        #print(n)
        return sum(n)

prob = Problem(1,5)

bestpop,cgcurve = GA(6,5,10,0.9,0.01,0.1,0.2,prob)
print(bestpop.Chromosome)
print(cgcurve)