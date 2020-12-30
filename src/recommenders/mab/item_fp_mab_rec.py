from cyclorec.recommenders.base_recommender import BaseRecommender
from cyclorec.mab.restriction_naive_mab import RestrictionNaiveMAB
from cyclorec.custom_exceptions.custom_exceptions import UnavailableArms, UserFullError
import math
import numpy as np


class FPItemMABRecommender(BaseRecommender):
    """
        Recommender that performs the recommendation task using a MAB whose arms are all the items in the recommender. It supports arm restriction in order
        to avoid an item to be recommended to the same user twice 
    """

    def __init__(self, dataLayer, mab_policy, epsilon=0.2, delta=2, alphas=0, betas=0):
        """ Constructor.

        Args:
            dataLayer ([type]): DataLayer over which recommendations will be performed.
            mab_policy (str): Indicates which algorithm the MAB uses to select which arm is selected. 
                              Must be one of:
                                    * greedy
                                    * e_greedy
                                    * random
                                    * thompson_sampling
                                    * ucb (upper confidence bound)
        """

        super().__init__(recommenderName='item_FP_MAB', dataLayer=dataLayer)  

        # Item MAB has the special restriction (in the environment of recommender systems) that it must not recommend 
        # the same item to the same user twice
        self.itemsDF = self.data_layer.get_items()
        self.mab = RestrictionNaiveMAB(policy=mab_policy, narms=self.itemsDF.shape[0], epsilon=epsilon, delta=delta, alphas=alphas, betas=betas)

       


    def select_item(self, userLayer, enable_repetitions=False):
        """ Chooses an item using the MAB. Then recommends it to the passed user. This is not a personalized recommender.

        Args:
            userLayer ([type]): UserLayer corresponding to the target user 
            enable_repetitions (bool, optional): True if it is possible to repeat recommendations. Defaults to False.

        Raises:
            UserFullError: The user can't be recommended any new item (only happens when enable_repetitions=False)

        Returns:
            pandas.Series -- contains the recommended item
        """
        idxs = userLayer.get_user_available_items_idxs(enable_repetitions)
        try:
            fake_index, self.last_armidx = self.mab.pull(available_arms=idxs)
            item_id = idxs[fake_index]
        except UnavailableArms:
            raise UserFullError

        return item_id

    def train(self):
        """ Method to implement the default training behaviour. It will only be executed once at the beginning of the recommendation task."""

        traindf = self.data_layer.get_initial_training_ratings_indexes()
        for (_, _, item_idx, rating) in traindf.itertuples():
            self.mab.pull_fixed_arm(item_idx)
            self.mab.update_rewards(item_idx, reward=rating, nanAsFailure=False)  
        return 
    
    def rating_update(self, user_idx, item_idx, rating):
        """ Updates the MAB estimation about the rewards associated to the previous arm. The header must be this way due to compatibility reasons.
        
        Arguments:
            user_id {int} --Not used
            item_id {int} -- Not used
            reward {int} -- Reward from last recommendation
        """
        reward = self.data_layer.get_reward(user_idx, item_idx, rating)        
        self.mab.update_rewards(self.last_armidx, reward, False)
        return 


    def epoch_update(self):
        """Defined for compatibility with BaseRecommender"""
        return 
            
