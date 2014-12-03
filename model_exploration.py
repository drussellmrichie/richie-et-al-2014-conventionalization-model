"""
This file runs a few different explorations of the Richie, Yang, & Coppola (2013)
topiCS model, using multicore processing. The top loops run through different 
values for different paramters. This will take a while if the number of 
parameters and number of values gets high.

For some reason, this won't run inside the Canopy IDE. Calling it from the 
terminal works, though.
"""

from multiprocessing import Pool
import itertools
from yang_et_al_2014_conventionalization_model import *
import csv

def pool_director(kwargs):
    return yang_convent_model(**kwargs)

n_cores = 8
pool    = Pool(n_cores)

sims_per_setting = 8

agents = [300]

conc_comp = [1]

objects = [1]

network = ['full','star']

#network = ['full','star','small-world']

#rewire_prob = [0,.001,.01,.1,1]

#comm_check = ['identical strings','expon decay']
                                                                                                                                            
arg_list = [{"agents": v, "conc_comp": w, "objects": x, "network": y} for v, w, x, y, _ in itertools.product(agents,conc_comp,objects,network,range(sims_per_setting))]

result_counter = 0

for result in pool.imap_unordered(pool_director, arg_list):
    print result_counter, result
    results_list.append(result)
    result_counter += 1

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results_list)
    