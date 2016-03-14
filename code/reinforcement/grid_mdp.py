'''
Created on Mar 13, 2016

@author: trucvietle
'''

from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
from textwrap import dedent
from itertools import *
import numpy as np

def parse():
    '''
    Parse command/terminal line.
    '''
    class CustomFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
        pass
    parser = ArgumentParser(formatter_class=CustomFormatter,
                            description=dedent('To be filled.'),
                            epilog=dedent('To be filled.'))
    parser.add_argument('-r', '--nrow', type=int, default=3, help='Number of rows')
    parser.add_argument('-c', '--ncol', type=int, default=4, help='Number of columns')
    return vars(parser.parse_args())
    
def build_environment(nrow=3, ncol=4):
    ## Set the initial rewards
    rewards = np.zeros(shape=(nrow, ncol))
    ## Coordinates of the 'wall'
    wall_r = int(nrow / 2)
    wall_c = int(ncol / 2) - 1
    ## Set the reward values
    rewards[wall_r, wall_c] = None
    rewards[0, ncol-1] = 1 # the goal
    rewards[1, ncol-1] = -1 # the trap
    
    ## Set the initial values
    values = rewards
    ## The set of states are the grid cells
    states = product(range(nrow), range(ncol))
#     for state in states:
#         print state
    ## Create the transition probabilities
    transition = {}
    transition['N'] = [0.8, 0, 0.1, 0.1] # north, south, east, west
    transition['S'] = [0, 0.8, 0.1, 0.1]
    transition['E'] = [0.1, 0.1, 0.8, 0]
    transition['W'] = [0.1, 0.1, 0, 0.8]
    
    ## The value of an action
    
    return rewards, values, states, transition
    
def act(action, state):
    '''
    Moves the agent through the states based on action taken.
    '''
    
if __name__ == '__main__':
    params = parse()
    nrow = params['nrow']
    ncol = params['ncol']
    build_environment(nrow, ncol)
    