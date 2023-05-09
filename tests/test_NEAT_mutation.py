from NEAT.genome import genome
from NEAT.components.node import node, Type
from NEAT.components.connection import connection

from NEAT.NEAT_Pool import NEAT_Pool

import numpy as np

def test_mutation():
    add_node_rate = 0
    add_connection_rate = 0
    adjust_weight_rate = 1

    pool = NEAT_Pool(3, 1, 2, genome,
                    add_node_rate,
                    add_connection_rate,
                    adjust_weight_rate)
    
    test_genome = pool.population[0]

    result_test_genome = pool.mutation(test_genome)
    print(test_genome.connection_genes[0].weight)
    print(result_test_genome.connection_genes[0].weight)

test_mutation()