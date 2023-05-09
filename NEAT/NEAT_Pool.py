from NEAT.genome import genome
from NEAT.components.node import node, Type
from NEAT.components.connection import connection

import numpy as np
import random
import copy

class NEAT_Pool:
    def __init__(self, num_inputs, 
                       num_outputs,
                       population_size,
                       genome_type = genome,
                       add_node_rate = 0.2,
                       add_connection_rate = 0.5,
                       adjust_weight_rate = 0.2,
                       num_elites = 2):

        #The population pool itself of genome objects
        self.population = []
        self.population_size = population_size

        #Number of input and output nodes in this pool
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        #Current generation
        self.generation = 0

        #Number of elite genomes to carry out from every generation
        self.num_elites = num_elites

        #Global innovation number counter
        self.curr_innovation_num = 0

        #Type of genome for this pool
        self.genome_type = genome_type

        #Initialize the population pool
        for i in range(0, population_size):
            new_genome = genome_type(num_inputs, num_outputs)
            new_genome.init_connection_genes(self)

            self.population.append(
                new_genome
            )

        #Rates related to the mutation operator
        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate
        self.adjust_weight_rate = adjust_weight_rate

    def fitness_function(self, genome):
        return 10*self.population.index(genome)+1
    
    def predict(self, X):
        if (X.shape != (self.population_size, self.num_inputs)):
            raise Exception('NEAT_Pool.predict: Make sure shape of X is ({0}, {1})'.format(
                str(self.population_size), str(self.num_inputs)
            ))
        
        Y = np.empty((self.population_size, self.num_outputs))

        #Getting output for all of the genomes
        for i in range(0, self.population_size):
            Y[i, :] = self.population[i].predict(X[i, :])
        
        return Y
    
    def reproduce(self):
        #Get the raw fitness values associated with each genome.
        #For easy access, pair fitness values and genomes together(key is fitness, value is genome)
        fitness_genome_pairing = {self.fitness_function(genome): genome for genome in self.population}

        #Sort the pairings based on fitness value
        fitness_genome_pairing = dict(sorted(fitness_genome_pairing.items()))

        #Getting all raw fitness values
        raw_fitnesses = list(fitness_genome_pairing.keys())

        #Getting sum of all the of raw fitness values
        total_fitnesses = sum(raw_fitnesses)
        
        #Adjusting fitness values to fitness/total_fitness for the selection operator
        fitness_genome_pairing = {
            (raw_fitness / (total_fitnesses)): fitness_genome_pairing[raw_fitness] 
            for raw_fitness in raw_fitnesses
        }

        #Initialize a new population pool, then get the elite individuals from last generation
        new_population_pool = []
        new_population_pool = new_population_pool + self.get_elite_genomes(fitness_genome_pairing)

        #For the remaining space in the new population size, select two parents,
        #breed them and apply mutations
        for i in range(self.num_elites, self.population_size):
            (parent1, fitness1) = self.select(fitness_genome_pairing)
            (parent2, fitness2) = self.select(fitness_genome_pairing)

            child = self.crossover(parent1, parent2, fitness1, fitness2)
            child = self.mutation(child)
            
            new_population_pool.append(child)
        
        self.population = new_population_pool

    def get_elite_genomes(self, fitness_genome_pairing):
        genomes = list(fitness_genome_pairing.values())
        return [genomes[i] for i in reversed(range(len(genomes)-self.num_elites, len(genomes)))]
            
    def select(self, fitness_genome_pairing):
        fitness_values = list(fitness_genome_pairing.keys())
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        chance_threshold = random.uniform(min_fitness, max_fitness)
        for fitness in fitness_values:
            if fitness >= chance_threshold:
                return (fitness_genome_pairing[fitness], fitness)

    def crossover(self, genome1, genome2, fitness1, fitness2):

        #Getting disjointed and matching connection genes based on the innovation numbers
        (disjoint_genes_1, disjoint_genes_2, joined_genes) = self.__find_disjoint_match_genes(genome1, genome2)
        new_genome = self.genome_type(self.num_inputs, self.num_outputs)

        #Inheriting matching genes
        for joined_gene in joined_genes:
            new_genome.connection_genes.add(random.choice(joined_gene))

        #Inheriting disjoint genes based on fitness
        if (fitness2 <= fitness1):
            for disjointed_gene in disjoint_genes_1:
                new_genome.connection_genes.add(disjointed_gene)
        if (fitness1 <= fitness2):
            for disjointed_gene in disjoint_genes_2:
                new_genome.connection_genes.add(disjointed_gene)
        
        curr_node_ids = [gene.id for gene in new_genome.node_genes]

        #Inherit node genes and randomly enable connection genes 25% of the time
        
        #Basically, given every in_node_id/out_node_id pairing in the connection genes,
        #if they don't yet exist in the new genome, create a new node gene for them
        for connection_gene in new_genome.connection_genes:
            if (not connection_gene.in_node_id in curr_node_ids):
                new_genome.node_genes.add(node(Type.Hidden, connection_gene.in_node_id))
                new_genome.curr_node_id += 1
                curr_node_ids.append(connection_gene.in_node_id)

            if (not connection_gene.out_node_id in curr_node_ids):
                new_genome.node_genes.add(node(Type.Hidden, connection_gene.out_node_id))
                new_genome.curr_node_id += 1
                curr_node_ids.append(connection_gene.out_node_id)

            #25% of the time, re-enable a connection gene
            if (not connection_gene.enabled and random.uniform(0, 1) < 0.25):
                connection_gene.enabled = True
        
        #Adjust the current node id based all of the recently inherited node genes
        new_genome.curr_node_id = max(curr_node_ids)+1

        return new_genome

    #Utilizing a 'merge' operation to find the disjoint and matching genes, as the connection genes are sorted based
    # on the innovation number
    def __find_disjoint_match_genes(self, genome1, genome2):
        disjoint_genes_1 = []
        disjoint_genes_2 = []
        joined_genes = []

        i = j = 0
        
        #Basically, 
        #if inno_num1 < inno_num2, a disjoint gene from genome1 is found,
        #if inno_num2 < inno_num1, a disjoint gene from genome2 was found
        #if inno_num1 == inno_num2, a matching gene was found
        while ((i < len(genome1.connection_genes)) and (j < len(genome2.connection_genes))) :
            if (genome1.connection_genes[i].innovation_number < genome2.connection_genes[j].innovation_number):
                disjoint_genes_1.append(genome1.connection_genes[i])
                i += 1

            elif (genome2.connection_genes[j].innovation_number < genome1.connection_genes[i].innovation_number):
                disjoint_genes_2.append(genome2.connection_genes[j])
                j += 1
            
            elif (genome1.connection_genes[i].innovation_number == genome2.connection_genes[j].innovation_number):
                joined_genes.append((genome1.connection_genes[i], genome2.connection_genes[j]))
                i += 1
                j += 1

        #Append the remaining disjoint genes
        while (i < len(genome1.connection_genes)):
            disjoint_genes_1.append(genome1.connection_genes[i])
            i += 1

        while (j < len(genome2.connection_genes)):
            disjoint_genes_2.append(genome2.connection_genes[j])
            j += 1

        return (disjoint_genes_1, disjoint_genes_2, joined_genes)
    
    def mutation(self, genome):
        mutated_genome = copy.deepcopy(genome)
        chance_threshold = random.uniform(0, 1)

        if (chance_threshold < self.add_node_rate):
            #Get a random connection gene
            connection_gene = random.choice(mutated_genome.connection_genes)

            #Add node gene to the genome
            new_node = node(Type.Hidden, mutated_genome.curr_node_id)
            mutated_genome.node_genes.add(new_node)

            #Adding connection genes inbetween the new node and the in/out ids of the selected connection gene
            #NOTE: the innovation counter is incremented to ensure connection genes can be matched up in the crossover
            #      operator in future generations
            first_connection_gene = connection(connection_gene.in_node_id,
                                               random.uniform(0, 1),
                                               mutated_genome.curr_node_id,
                                               self.curr_innovation_num)
            self.curr_innovation_num += 1

            second_connection_gene = connection(mutated_genome.curr_node_id,
                                               random.uniform(0, 1),
                                               connection_gene.out_node_id,
                                               self.curr_innovation_num)
            self.curr_innovation_num += 1
            mutated_genome.connection_genes.add(first_connection_gene)
            mutated_genome.connection_genes.add(second_connection_gene)

            #Disabling the selected connection gene
            connection_gene.enabled = False

            #Increment the node id to accompany for new nodes in future generations
            mutated_genome.curr_node_id += 1
            
        if (chance_threshold < self.add_connection_rate):

            #Getting all input_ids, output_ids, and the ids as a whole
            input_ids = [curr_node.id for curr_node in mutated_genome.node_genes if curr_node.type == Type.Input]
            output_ids = [curr_node.id for curr_node in mutated_genome.node_genes if curr_node.type == Type.Output]
            all_ids = [curr_node.id for curr_node in mutated_genome.node_genes]            
            
            #Selecting an input/output id pair, 
            # while ensuring the selected input/output id is not within the output/input id pool
            selected_in_id = random.choice(all_ids)
            while (selected_in_id in output_ids):
                selected_in_id = random.choice(all_ids)
            
            selected_output_id = random.choice(all_ids)
            while (selected_output_id in input_ids):
                selected_output_id = random.choice(all_ids)
            
            #Add the connection gene to the genome, then increment the innovation number counter
            new_connection_gene = connection(
                selected_in_id,
                random.uniform(0, 1),
                selected_output_id,
                self.curr_innovation_num
            )

            mutated_genome.connection_genes.add(new_connection_gene)
            self.curr_innovation_num += 1

        if (chance_threshold < self.adjust_weight_rate):
            #Get a random connection gene
            connection_gene = random.choice(mutated_genome.connection_genes)

            #Adjust the weight to a random number
            connection_gene.weight = random.uniform(0, 1)

        return mutated_genome