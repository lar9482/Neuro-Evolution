from NEAT.genome import genome
from NEAT.components.node import node, Type
from NEAT.components.connection import connection

from NEAT.NEAT_Pool import NEAT_Pool

import numpy as np

#Emulating the genomes in Stanley's crossover example
def get_stanley_genomes():
    num_inputs = 3
    num_outputs = 1
    innovation_number = 0

    #Initializing the genomes
    first_genome = genome(num_inputs, num_outputs)
    second_genome = genome(num_inputs, num_outputs)

    #Adding the nodes
    first_genome.node_genes.add(node(Type.Hidden, first_genome.curr_node_id))
    first_genome.curr_node_id += 1

    second_genome.node_genes.add(node(Type.Hidden, second_genome.curr_node_id))
    second_genome.curr_node_id += 1

    second_genome.node_genes.add(node(Type.Hidden, second_genome.curr_node_id))
    second_genome.curr_node_id += 1

    #Adding connections
    #1st set
    first_genome.connection_genes.add(
        connection(0, 0.5, 3, innovation_number)
    )
    second_genome.connection_genes.add(
        connection(0, 0.5, 3, innovation_number)
    )
    innovation_number+=1

    #2nd set
    conn1 = connection(1, 0.5, 3, innovation_number)
    conn1.enabled = False
    first_genome.connection_genes.add(
        conn1
    )
    conn2 = connection(1, 0.5, 3, innovation_number)
    conn2.enabled = False
    second_genome.connection_genes.add(
        conn2
    )
    innovation_number+=1

    #3rd set
    first_genome.connection_genes.add(
        connection(2, 0.5, 3, innovation_number)
    )
    second_genome.connection_genes.add(
        connection(2, 0.5, 3, innovation_number)
    )
    innovation_number+=1

    #4th set
    first_genome.connection_genes.add(
        connection(1, 0.5, 4, innovation_number)
    )
    second_genome.connection_genes.add(
        connection(1, 0.5, 4, innovation_number)
    )
    innovation_number+=1

    #5th set
    first_genome.connection_genes.add(
        connection(4, 0.5, 3, innovation_number)
    )
    second_genome.connection_genes.add(
        connection(4, 0.5, 3, innovation_number)
    )
    innovation_number+=1

    #6th set
    second_genome.connection_genes.add(
        connection(4, 0.5, 5, innovation_number)
    )
    innovation_number+=1

    #7th set
    second_genome.connection_genes.add(
        connection(5, 0.5, 3, innovation_number)
    )
    innovation_number+=1

    #8th set
    first_genome.connection_genes.add(
        connection(0, 0.5, 4, innovation_number)
    )
    innovation_number+=1

    #9th set
    second_genome.connection_genes.add(
        connection(2, 0.5, 4, innovation_number)
    )
    innovation_number+=1

    #10th set
    second_genome.connection_genes.add(
        connection(0, 0.5, 5, innovation_number)
    )
    innovation_number+=1

    return (first_genome, second_genome)


def first_test():

    (first_genome, second_genome) = get_stanley_genomes()
    pool = NEAT_Pool(3, 1, 2, genome)

    child_genome = pool.crossover(first_genome, second_genome, 1, 1)

    X = np.array([1, 2, 3], np.int32)
    Y = child_genome.predict(X)

    print(X)
    print(Y)

first_test()