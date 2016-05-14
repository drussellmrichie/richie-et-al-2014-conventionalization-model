"""
READ ME

This is a re-implementation of the agent-based model of conventionalization in
Richie, Yang, & Coppola (2013, topiCS).

"""

import os
import random
import numpy
import networkx
import math

os.chdir('/Users/russellrichie/yang_convent_model')

def flip(p):
    return 1 if random.random() < p else 0

def hamming_distance(s1, s2):
    "Return the Hamming distance between equal-length sequences."
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))    
            
def update((cc, p, gamma)): #it has to take a single tuple argument to deal with the zipped data
    if cc == 1:
        new_p = p + gamma*(1 - p)
    else:
        new_p = (1 - gamma)*p
    return new_p
    
def check_convent(nd_array,conv_crit): # there *has* to be a more efficient way to check if convent has happened yet...
    for prob in nd_array.ravel():
        if (prob < (1 - conv_crit)) & (prob > conv_crit):
            return True
    return False

#def betweenness_centralization(G):                # the algorithm I found online
#    vnum = networkx.number_of_nodes(G)
#    if vnum < 3:
#        raise ValueError("graph must have at least three vertices")
#    denom = (vnum-1)*(vnum-2)
#    betweenness_dict = networkx.betweenness_centrality(G)
#    temparr = [2*i/denom for i in betweenness_dict.values()]
#    max_temparr = max(temparr)
#    return sum(max_temparr-i for i in temparr)/(vnum-1)
#
#def my_betweenness_centralization(graph):      # the algorithm I wrote
#    vnum = networkx.number_of_nodes(graph)
#    if vnum < 3:
#        raise ValueError("graph must have at least three vertices")
#    betweenness_dict = networkx.betweenness_centrality(graph)
#    max_bet = max(betweenness_dict.values())
#    numerator = sum((x-max_bet) for x in betweenness_dict.values())
#    return numerator/(vnum-1)

def yang_convent_model(agents = 5, 
                        objects = 1,
                        conc_comp = 40, 
                        network = 'full',
                        rewire_prob = 0, 
                        comm_check = 'expon decay',
                        neighbors = 10,
                        probs = 'uniform',
                        gamma = .01,
                        conv_crit = .02,
                        simul_conv = 'no',
                        no_convos = 2000000,
                        data_intervals = 'no' # may just want to change the code if actually want to use data_intervals?
                        ):
    """
    The agent-based conventionalization model of Richie, Yang, and Coppola (2013, topiCS).
    """
    #print agents, objects, conc_comp, network, rewire_prob, comm_check # add other parameters here
    
    # Initialize the agents' probabilities    
    
    if probs == '.9 or .1':
        listed_matrix = [random.choice((.9,.1)) for x in range(agents*objects*conc_comp)]
        prob_matrix = numpy.reshape(listed_matrix,newshape=(agents,objects,conc_comp))
    elif probs == 'uniform':
        prob_matrix = numpy.random.random((agents,objects,conc_comp))
    elif probs == '.5':
        prob_matrix = numpy.zeros(shape=(agents,objects,conc_comp)) + .5
    
    # Intialize the social network
    
    if network == 'full':
        social_network = networkx.complete_graph(agents)
    elif network == 'star':
        social_network = networkx.star_graph(agents-1)
    elif network == 'small-world':
        social_network = networkx.connected_watts_strogatz_graph(agents, neighbors, rewire_prob, tries=100, seed=None)
    # can add other network possibilities here
    
    neighbors = [social_network.neighbors(x) for x in range(agents)] # generate the list of neighbors which will be used for picking listeners
                            
    # Run the conversations
    if comm_check == 'expon decay':
        for convo_ind in xrange(0,no_convos):
            print convo_ind
            
            speaker_ind = random.randint(0, agents-1) # pick a speaker
            listener_ind = random.choice(neighbors[speaker_ind]) # pick a listener based on who speaker can talk to
            object_ind = random.randint(0, objects-1) # pick an object
            
            speaker_string = [flip(x) for x in prob_matrix[speaker_ind, object_ind, : ]] # speaker utters a string
            listener_string = [flip(x) for x in prob_matrix[listener_ind, object_ind, : ]] # listener utters a string

            distance = hamming_distance(speaker_string,listener_string) # hamming distance between speaker's and listener's strings
            comm_success_prob = math.e**(-distance) # probability of successful communication is exponential decay function of hamming distance

            if flip(comm_success_prob) == 1: # if communication was successful, then update listener, and check if convent has happened
                             
                #for ind, bit in enumerate(speaker_string):
                #    if bit == 1:
                #        prob_matrix[listener_ind,object_ind, ind] = prob_matrix[listener_ind,object_ind, bit] + gamma*(1 - prob_matrix[listener_ind,object_ind, ind])
                #    else:
                #        prob_matrix[listener_ind,object_ind, ind] = (1 - gamma) * prob_matrix[listener_ind,object_ind, ind]
                #
                prob_matrix[listener_ind, object_ind, : ] = map(update, zip(speaker_string, prob_matrix[listener_ind, object_ind, :], [gamma]*len(speaker_string))) #update probabilities
                
                if check_convent(prob_matrix,conv_crit): # check if all probs are still outside of crit value
                    continue
                else:
                    break
            
            else: # if communication was not successful, then continue to next conversation
                continue
    if comm_check == 'identical strings':
        for convo_ind in xrange(0,no_convos):
            #print convo_ind

            speaker_ind = random.randint(0, agents-1) # pick a speaker
            listener_ind = random.choice(social_network.neighbors(speaker_ind)) # pick a listener based on who speaker can talk to
            object_ind = random.randint(0, objects-1) # pick an object
            
            speaker_string = [flip(x) for x in prob_matrix[speaker_ind, object_ind, : ]] # speaker utters a string
            listener_string = [flip(x) for x in prob_matrix[listener_ind, object_ind, : ]] # listener utters a string
            
            if speaker_string == listener_string: # if communication was successful, then update listener, and check if convent has happened
                             
                #for ind, bit in enumerate(speaker_string):
                #    if bit == 1:
                #        prob_matrix[listener_ind,object_ind, ind] = prob_matrix[listener_ind,object_ind, bit] + gamma*(1 - prob_matrix[listener_ind,object_ind, ind])
                #    else:
                #        prob_matrix[listener_ind,object_ind, ind] = (1 - gamma) * prob_matrix[listener_ind,object_ind, ind]
                #
                prob_matrix[listener_ind, object_ind, : ] = map(update, zip(speaker_string, prob_matrix[listener_ind, object_ind, :], [gamma]*len(speaker_string))) #update probabilities
                
                if check_convent(prob_matrix,conv_crit): # check if all probs are still outside of crit value
                    continue
                else:
                    break
            
            else: # if communication was not successful, then continue to next conversation
                continue
    return [agents, 
            objects, 
            conc_comp,
            network, 
            convo_ind, 
            networkx.average_clustering(social_network), 
            networkx.average_shortest_path_length(social_network)]
            #betweenness_centralization(social_network)]
    """
    return {'convo_ind': convo_ind, 
            'agents': agents, 
            'objects':objects, 
            'conc_comp':conc_comp, 
            'network':network, 
            'rewire_prob':rewire_prob, 
            'comm_check':comm_check} #prob_matrix #convo_ind should be the index of the last converation, either the one by which convent happened, or no_convos
    """