import numpy as np
import time
import math

# OWN
from cyclorec.mab.naive_mab import NaiveMAB
from cyclorec.mab.mab_policies import greedy, e_greedy, random, ts, ucb
from cyclorec.custom_exceptions.custom_exceptions import UnavailableArms



class RestrictionNaiveMAB(NaiveMAB):
    """ This class generalizes the NaiveMAB problem assuming there are "forbbiden arms" at each iteration. """

    def __init__(self, policy, narms, epsilon, delta, alphas, betas):
        """ Constructor

        Args:
            policy (str): Must be one of ['greedy', 'e_greedy', 'random', 'ts', 'ucb']. Algorithm to choose between arms.
            narms (int): Number of arms to be created.
        """

        super().__init__(policy, narms, epsilon, delta, alphas, betas) 


    def pull(self, available_arms):
        """Override the default MAB behaviour. It is required because we must not recommend the same item to an user twice. Therefore, in an ItemMAB
           each user has a list of arms that can't be pulled again(chosen)
        
        Arguments:

        
        Keyword Arguments:
            enable_repetitions {bool} -- [description] (default: {False})
        
        Returns:
            [type] -- [description]
        """ 

        # We have to filter the values passed to the MAB excluding the items which have been selected yet
        #available_arms = [index for index, _ in enumerate(self.reward_estimates) if index not in forbidden_arms]
        #available_arms = np.array(available_arms)
        available_rewards = self.reward_estimates[available_arms]
        available_alphas =  self.alphas[available_arms]
        available_betas = self.betas[available_arms]
        available_n_pulls = self.num_pulls[available_arms]
        fake_armidx = self.policy(rewards=available_rewards, n_pulls=available_n_pulls, alphas=available_alphas, betas=available_betas, mab_epoch=self.t, epsilon=self.epsilon, delta=self.delta)

        # Index pullback
        armidx = available_arms[fake_armidx]
        self.num_pulls[armidx] += 1
        self.t += 1
        return fake_armidx, armidx

    
