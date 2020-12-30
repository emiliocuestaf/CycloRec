import numpy as np
import math
# OWN
from cyclorec.mab.mab_policies import greedy, e_greedy, random, ts, ucb
from cyclorec.custom_exceptions.custom_exceptions import UserFullError



class NaiveMAB:
    """
        This is a Multi Armed Bandit (MAB) where the arm pulling action is done outside the proper bandit. This class acts solely as an 'option choser'.
        In order to work properly, it is important to call the update_rewards() method manually, every time an arm is pulled.
    """


    def __init__(self, policy, narms, epsilon=0.2, delta=2, alphas=0, betas=0):
        """ Constructor

        Args:
            policy (str): Must be one of ['greedy', 'e_greedy', 'random', 'ts', 'ucb']. Algorithm to choose between arms.
            narms (int): Number of arms to be created.
            epsilon (float): Probability used in epsilon-greedy policy. Must be between 0 and 1. 
            delta (float): Delta for Upper Confidence Bound policy. Must be > 0
            alphas (float): Initial value of alphas. Must be >= 0. 
            betas (float): Initial value of betas. Must be >= 0.
        """

        valid_policies = ['greedy', 'e_greedy', 'random', 'ts', 'ucb']
        assert (policy in valid_policies), "{} MAB policy does not exist. Try with any of {}".format(policy, valid_policies)
        self.policy = eval(policy)        


        # Indicates the number of times each arm has been pulled
        self.num_pulls = np.array([0]*narms)

        # Alphas and Betas (used in TS for example)
        self.alphas = np.array([alphas]*narms)
        self.alpha0 = alphas
        self.betas = np.array([betas]*narms)
        self.beta0 = betas

        # Arms and reward_estimates are index related, this means the estimated reward for arms[i] is stored in reward_estimates[i]
        assert (self.alpha0 >=0 and self.beta0>=0), "Alphas and Betas must be both >= 0"
        if self.alpha0 == 0:
            self.init_estimate = 0
        elif self.alpha0 > 0 and self.beta0 >= 0:
            self.init_estimate = (self.alpha0)/(self.alpha0 + self.beta0)
        self.reward_estimates = np.array([float(self.init_estimate)]*narms)

        # Params for other algorithms
        self.epsilon = epsilon # E-greedy
        self.delta = delta # UCB
        
        # Sum of rewards
        self.totalrewards = 0
        
        # Number of iterations
        self.t = 1


    def pull(self):
        """ Chooses an arm using the defined policy, updates the historical registers and returns its index

        Returns:
            armidx (int): Arm which was pulled
        """
        armidx = self.policy(rewards=self.reward_estimates, n_pulls=self.num_pulls, alphas=self.alphas, betas=self.betas, mab_epoch=self.t, epsilon=self.epsilon, delta=self.delta)
        # In this case, the armidx returned by the mab policy is coherent with the whole MAB indexation. 
        self.num_pulls[armidx] += 1
        self.t += 1
        return armidx
    

    def pull_fixed_arm(self, armidx):
        """ Pulls an arm updating its historical registers

        Args:
            armidx (int): Arm to be pulled
        """
        self.num_pulls[armidx] += 1
        self.t += 1
        return 

    
    def update_rewards(self, armidx, reward, nanAsFailure=True):
        """ Updates the reward-value estimation for the passed arm.

        Args:
            armidx (int): Arm to be updated
            reward (float): Associated 
        """
        if not math.isnan(reward) and reward == 1: # Success
            #self.reward_estimates = self.reward_estimates * self.num_pulls
            self.alphas[armidx] += 1
            self.reward_estimates[armidx] = self.alphas[armidx]/(self.alphas[armidx]+self.betas[armidx])
            self.totalrewards += reward
        elif not math.isnan(reward)  and reward == 0: #Failure
            self.betas[armidx] += 1
            self.reward_estimates[armidx] = self.alphas[armidx]/(self.alphas[armidx]+self.betas[armidx])
        elif nanAsFailure: #Failure
            self.betas[armidx] += 1
            self.reward_estimates[armidx] = self.alphas[armidx]/(self.alphas[armidx]+self.betas[armidx])

        #self.reward_estimates[armidx] = (self.reward_estimates[armidx]*(self.num_pulls[armidx] - 1) + reward)/(self.num_pulls[armidx])
        return 


    def reset_bandit(self):
        """ Resets bandit to an initial stage """
        self.totalrewards = 0
        self.num_pulls.fill(0)
        self.t = 1

        # Reward estimations
        self.alphas.fill(self.alpha0)
        self.betas.fill(self.beta0)
        self.reward_estimates.fill(self.init_estimate)
        return 



