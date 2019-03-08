# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: QUVA Lab
# Date Created: 2019-02-27

import sys
import time
import tqdm

import numpy as np
import torch
from torch.autograd import Variable

################################################################################
################################################################################

'''

Usage to reproduce the setting of Snoek et al. - "Practical Bayesian Optimization":

python HyperSphere/BO/run_BO.py \
    --geometry cube \
    --ard --parallel \
    --func FUNCTION_TO_EVALUATE \
    --dim DIMENSIONALITY \
    --n_eval NUMBER_OF_EVALUATIONS

EXAMPLE:
    python HyperSphere/BO/run_BO.py --geometry cube --ard --parallel --func arcsim_simulation --n_eval 10

'''

num_bending_params = 4
num_params = num_bending_params + 1
    
def arcsim_simulation(config_vector):

    labels = ['bending_{}'.format(i) for i in range(num_bending_params)]
    labels.append('wind_speed')

    if len(config_vector) != len(labels):
        raise ValueError('Length of configuration vector does not match labels.')

    # Extract configuration into a dictionary
    config = config_vector.data.numpy().tolist()
    config = dict(list(zip(labels, config)))
    
    loss = np.random.uniform(0, 5)
    print('Loss = {:.3f}'.format(loss))

    return Variable(torch.FloatTensor([[loss]]))

# Manually specify the dimensionality
arcsim_simulation.dim = num_params

