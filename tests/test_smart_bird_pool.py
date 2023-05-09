from NEAT.NEAT_Pool import NEAT_Pool
from smart_entities.smart_bird import smart_bird
from smart_entities.smart_bird_pool import smart_bird_pool

import numpy as np

def test_smart_bird_pool():
    screen_size = (500, 700)
    pool = smart_bird_pool(screen_size, population_size=2)

    pool.population[0].start_time = 0
    pool.population[0].end_time = 10

    pool.population[1].start_time = 0
    pool.population[1].end_time = 20
    
    x = np.array([[1, 2, 3], 
              [5, 6, 7]], np.int32)

    print(pool.predict(x))
    print()
    pool.reproduce()
    print(pool.predict(x))

test_smart_bird_pool()