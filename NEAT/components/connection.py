
class connection:
    def __init__(self, in_node_id,  weight, out_node_id, innovation_number):
        self.in_node_id = in_node_id
        self.weight = weight
        self.out_node_id = out_node_id

        self.enabled = True
        
        self.innovation_number = innovation_number

    #Less than 'operator'
    def __lt__(self, other):
        return self.innovation_number <= other.innovation_number
    