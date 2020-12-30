import numpy as np
import pandas as pd
import math
#from mab.item_mab import ItemMAB

from cyclorec.custom_exceptions.custom_exceptions import UnavailableArms


""" All these functions are made to perform a choice between the arms of a Multi-Armed Bandit (MAB)"""
""" They must have the SAME ARGUMENTS due to compatibility reasons"""

def random(rewards=None, n_pulls=None, alphas=None, betas=None, mab_epoch=None, epsilon=None, delta=None):
    """ Selects an arm randomly.

    Args:
        rewards ([int], optional): [Reward estimations for each arm (index related)]. Defaults to None.
        n_pulls ([int], optional): [List/array containing the number of pulls of each arm (index related)]. Defaults to None.
        alphas ([int], optional): [Number of success for each arm (index related)]. Defaults to None.
        betas ([int], optional): [Number of failures for each arm (index related)]. Defaults to None.
        mab_epoch (int, optional): Ignored. Must be positive Defaults to None.
        epsilon (float, optional): [Ignored]. Defaults to None.
        delta (float, optional): [Ignored]. Defaults to None.

    Raises:
        UnavailableArms: Thrown when there are no arms to choose. For example, when a user has been recommended all the items.

    Returns:
        int: Index of the selected arm
    """
    if len(rewards) == 0:
        raise UnavailableArms()

    return np.random.randint(len(rewards))
    

def greedy(rewards=None, n_pulls=None, alphas=None, betas=None, mab_epoch=None, epsilon=None, delta=None):
    """ Selects an arm using a greedy policy (always chooses the item with the best estimated reward)

    Args:
        rewards ([int], optional): [Reward estimations for each arm (index related)]. Defaults to None.
        n_pulls ([int], optional): [List/array containing the number of pulls of each arm (index related)]. Defaults to None.
        alphas ([int], optional): [Number of success for each arm (index related)]. Defaults to None.
        betas ([int], optional): [Number of failures for each arm (index related)]. Defaults to None.
        mab_epoch (int, optional): Ignored. Must be positive Defaults to None.
        epsilon (float, optional): [Ignored]. Defaults to None.
        delta (float, optional): [Ignored]. Defaults to None.

    Raises:
        UnavailableArms: Thrown when there are no arms to choose. For example, when a user has been recommended all the items.

    Returns:
        int: Index of the selected arm
    """   
    if len(rewards) == 0:
        raise UnavailableArms()

    max_idxs = np.where(rewards == np.max(rewards))[0]
    return np.random.choice(max_idxs)    



def e_greedy(rewards=None, n_pulls=None, alphas=None, betas=None, mab_epoch=None, epsilon=None, delta=None):
    """ Selects an arm using an egreedy policy. With \epsilon probability, the algorihtm chooses an arm randomly (explore), 
        otheriwise, it chooses the item that maximizes the estimated reward. 

    Args:
        rewards ([int], optional): [Reward estimations for each arm (index related)]. Defaults to None.
        n_pulls ([int], optional): [List/array containing the number of pulls of each arm (index related)]. Defaults to None.
        alphas ([int], optional): [Number of success for each arm (index related)]. Defaults to None.
        betas ([int], optional): [Number of failures for each arm (index related)]. Defaults to None.
        mab_epoch (int, optional): Ignored. Must be positive Defaults to None.
        epsilon (float, optional): [Epsilon used for the draw.]. Defaults to None.
        delta (float, optional): [Ignored]. Defaults to None.

    Raises:
        UnavailableArms: Thrown when there are no arms to choose. For example, when a user has been recommended all the items.

    Returns:
        int: Index of the selected arm
    """
    if len(rewards) == 0:
        raise UnavailableArms()
    
    draw = np.random.random_sample()    
    if draw < epsilon:
        # Exploration branch (random)
        fake_armidx = np.random.randint(len(rewards))
    else:
        # Exploitation branch
        max_idxs = np.where(rewards == np.max(rewards))[0]
        fake_armidx = np.random.choice(max_idxs)

    # Be careful with the returned index, if the passed arrays are not the same size than the MABs arm array,
    # the indexing could be incoherent. This is useful to deal with "forbidden" arms at each time t or for each user.
    return fake_armidx


def ts(rewards=None, n_pulls=None, alphas=None, betas=None, mab_epoch=None, epsilon=None, delta=None):
    """ Selects an arm performing a Thompson Sampling draw. All the arms have an associated Beta distribution.
        This distribution is drawn for each arm and the arm with the biggest value is pulled.

    Args:
        rewards ([int], optional): [Reward estimations for each arm (index related)]. Defaults to None.
        n_pulls ([int], optional): [List/array containing the number of pulls of each arm (index related)]. Defaults to None.
        alphas ([int], optional): [Number of success for each arm (index related)]. Defaults to None.
        betas ([int], optional): [Number of failures for each arm (index related)]. Defaults to None.
        mab_epoch (int, optional): Ignored. Defaults to None.
        epsilon (float, optional): [Ignored]. Defaults to None.
        delta (float, optional): [Ignored]. Defaults to None.

    Raises:
        UnavailableArms: Thrown when there are no arms to choose. For example, when a user has been recommended all the items.

    Returns:
        int: Index of the selected arm
    """
    if len(rewards) == 0:
        raise UnavailableArms()

    # FIRST CASE: TS needs all the arms to have been pulled at least once, so this is the first step (Ensure everything has been pulled)
    # zeroes_indexes = np.where(n_pulls == 0)[0]
    # if len(zeroes_indexes) > 0:
    #     return np.random.choice(zeroes_indexes)
    
    # SECOND CASE: Actual TS
    else:
        # Drawing from beta distribution using only the passed alphas and betas
        draws = np.random.beta(alphas+1, betas+1)
        max_idxs = np.where(draws == np.max(draws))[0]
        fake_armidx = np.random.choice(max_idxs)
        return fake_armidx

    
 

def ucb(rewards=None, n_pulls=None, alphas=None, betas=None, mab_epoch=None, epsilon=None, delta=None):
    """ Selects an arm using an UCB1 policy. 

    Args:
        rewards ([int], optional): [Reward estimations for each arm (index related)]. Defaults to None.
        n_pulls ([int], optional): [List/array containing the number of pulls of each arm (index related)]. Defaults to None.
        alphas ([int], optional): [Number of success for each arm (index related)]. Defaults to None.
        betas ([int], optional): [Number of failures for each arm (index related)]. Defaults to None.
        mab_epoch (int, optional): epoch. Must be positive Defaults to None.
        epsilon (float, optional): Ignored. Defaults to None.
        delta (float, optional): delta value used to calculate the exploration term. Defaults to None.

    Raises:
        UnavailableArms: Thrown when there are no arms to choose. For example, when a user has been recommended all the items.

    Returns:
        int: Index of the selected arm
    """
    if len(rewards) == 0:
        raise UnavailableArms()

    # FIRST CASE: UCB needs all the arms to have been pulled at least once, so this is the first step (Ensure everything has been pulled)
    zeroes_indexes = np.where(n_pulls == 0)[0]
    if len(zeroes_indexes) > 0:
        return np.random.choice(zeroes_indexes)
    
    # SECOND CASE: Actual UCB
    else:
        if delta < 0:
            # Patch: Negative delta indicates its value must increase over time. Otherwise its value will be constant.  
            delta = 1 + mab_epoch * pow((math.log(mab_epoch)), 2)

        # Calculating the final estimation value for each of the arms (averages exploration and exploitation)
        values = rewards +  np.sqrt(delta*math.log(mab_epoch)/n_pulls)

        max_idxs = np.where(values == np.max(values))[0]
        fake_armidx =  np.random.choice(max_idxs)
        return fake_armidx



 