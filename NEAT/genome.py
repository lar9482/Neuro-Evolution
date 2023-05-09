from sortedcontainers import SortedList
from NEAT.components.node import node, Type
from NEAT.components.connection import connection
import random

import numpy as np

class genome:
    def __init__(self, num_inputs, num_outputs):

        #Number of input and output nodes in this genome
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        #The pool of node genes and connection genes
        #They are sorted based on the node IDs/innovation numbers
        self.node_genes = SortedList()
        self.connection_genes = SortedList()

        #Node ID counter for this genome.
        self.curr_node_id = 0
        
        #Insert all input/output nodes into node gene pool
        self.__init_node_genes(num_inputs, num_outputs)

    def __init_node_genes(self, num_inputs, num_outputs):

        #Given the number of inputs, add input nodes with unique node ids
        for i in range(0, num_inputs):
            self.node_genes.add(node(Type.Input, i))
            self.curr_node_id += 1

        #Given the number of output, add output nodes with unique node ids
        for i in range(self.num_inputs, self.num_inputs+num_outputs):
            self.node_genes.add(node(Type.Output, i))
            self.curr_node_id += 1

    def init_connection_genes(self, NEAT_Pool):
        chosen_input_id = random.choice(list(range(0, self.num_inputs)))
        chosen_output_id = random.choice(list(range(self.num_inputs, self.num_inputs+self.num_outputs)))

        new_connection_gene = connection(chosen_input_id,
                                         random.uniform(0, 1),
                                         chosen_output_id,
                                         NEAT_Pool.curr_innovation_num)
        
        self.connection_genes.add(new_connection_gene)
        NEAT_Pool.curr_innovation_num += 1

    def __reset_nodes(self):
        #Set all value/activated_values from all of the nodes to 0
        for node in self.node_genes:
            node.reset_node()

    def __feed_in(self, input):
        #Getting the initial input node ids
        input_ids = [curr_node.id for curr_node in self.node_genes if curr_node.type == Type.Input]

        #For every input id, set the associated node's value to the input passed in.
        for input_id in input_ids:
            self.node_genes[input_id].value = input[input_id]

    def __feed_forward(self):

        #Initialize the current ids tracked to the input ids
        #Treat as a queue to process node ids one by one
        curr_ids = [curr_node.id for curr_node in self.node_genes if curr_node.type == Type.Input]

        #Getting the target ids, which are the output node ids.
        output_ids = [curr_node.id for curr_node in self.node_genes if curr_node.type == Type.Output]

        while (len(curr_ids) != 0):

            #Dequeue a node id, then activating the node associated with the id.
            curr_id = curr_ids.pop(0)
            self.node_genes[curr_id].activate_node()

            #Scanning through all of the connections
            for conn in self.connection_genes:

                #Given a connection that has the input id as the current id
                if (conn.in_node_id == curr_id and conn.enabled):
                    
                    #Take the weight and current id, and data into the node associated with the output id
                    self.node_genes[conn.out_node_id].feed(
                        conn.weight, self.node_genes[curr_id].activated_value
                    )

                    #Given the out_node_id isn't an output id, add it to the queue
                    if (not conn.out_node_id in output_ids 
                        and conn.out_node_id != curr_id
                        and conn.out_node_id in curr_ids
                        ):
                        curr_ids.append(conn.out_node_id)

    def __get_output(self):
        #Getting the output ids
        output_ids = output_ids = [curr_node.id for curr_node in self.node_genes if curr_node.type == Type.Output]

        output = np.empty(self.num_outputs, np.float32)

        #For all of the output ids, activate the node and get the activated value
        for i in range(0, len(output_ids)):
            self.node_genes[output_ids[i]].activate_node()
            output[i] = self.node_genes[output_ids[i]].activated_value

        return output 


    def predict(self, input):
        if (len(input) != self.num_inputs):
            raise Exception('genome.feed_forward: Make sure input shape is equal to the number of inputs')

        #Clear all values/activated_values from the node genes
        self.__reset_nodes()

        #Placing the input data into the input nodes initially
        self.__feed_in(input)

        #Feed the data through the genes into the output nodes
        self.__feed_forward()
        
        #Getting the activated data from the output nodes
        return self.__get_output()