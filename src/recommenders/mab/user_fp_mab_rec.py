import numpy as np

from cyclorec.recommenders.base_recommender import BaseRecommender
from cyclorec.mab.naive_mab import NaiveMAB
from cyclorec.custom_exceptions.custom_exceptions import UnavailableArms, UserFullError


class FPUserMABRecommender(BaseRecommender):
    """
        Recommender that performs the recommendation task using a MAB whose arms are all the users in the recommender.
    """


    def __init__(self, dataLayer, mab_policy, epsilon=0.2, delta=2, alphas=1, betas=1):
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

        super().__init__(recommenderName='user_FP_MAB', dataLayer=dataLayer)  
   
        # MAB Creation
        self.users_list = self.data_layer.list_users()
        self.mab = NaiveMAB(policy=mab_policy, narms=len(self.users_list), epsilon=epsilon, delta=delta, alphas=alphas, betas=betas)
       
    def select_item(self, userLayer, enable_repetitions=False):
        """ Chooses a neighbor user using the MAB, sorts the ratings of the neighbor user and recommends the first one that has 
            not been recommended to the target user yet (if enable_repetitions = False)

        Args:
            userLayer ([type]): userLayer associated to the user.
            enable_repetitions (bool, optional): True if it is possible to repeat recommendations. Defaults to False.

        Raises:
            UserFullError: The user can't be recommended any new item (only happens when enable_repetitions=False)

        Returns:
            pandas.Series -- contains the recommended item
        """
        
        #user = userLayer.get_user()
        eligible_items = userLayer.get_user_available_items_idxs(enable_repetitions)
        
        if eligible_items.shape[0] == 0:
            raise UserFullError

        self.last_armidx = self.mab.pull()
        neighbor_id = self.users_list[self.last_armidx]
        rec_items = self.data_layer.get_user_known_ratings(neighbor_id)

        if rec_items.shape[0] == 0:
            items = eligible_items.sample(n = 1, replace=False)
            item = items.iloc[0,:]
            return item

        merged_items = eligible_items.merge(right=rec_items, how='inner', left_index=True, right_index=True)

        if merged_items.shape[0] == 0:
            items = eligible_items.sample(n = 1, replace=False)
            item = items.iloc[0,:]
            return item

        item = merged_items.loc[merged_items['rating'].idxmax()]
        return item

    def train(self):
        """ Method to implement the default training behaviour. It will only be executed once at the beginning of the recommendation task."""

        traindf= self.data_layer.get_initial_training_ratings()
        

        for (_, user_id, _, rating) in traindf.itertuples():

            user_idx = self.data_layer.get_users().index.get_loc(user_id)
            self.mab.pull_fixed_arm(user_idx)
            self.mab.update_rewards(user_idx, reward=rating)
        
        return 


    def rating_update(self, user_id, item_id, rating):
        """ Updates the MAB estimation about the rewards associated to the previously chosen arm. The header must be this way due to compatibility reasons.
        
        Arguments:
            user_id {int} --Not used
            item_id {int} -- Not used
            reward {int} -- Reward from last recommendation
        """

        # This, below, were the conditions of the normal bandit recommender. In this approach, we try to minimize failure, so only 
        # total failed recommendations will be penalized.

        #if reward != 0 and not np.isnan(reward):
        #    self.mab.update_rewards(self.last_armidx, reward)

        reward = self.data_layer.get_bandit_reward(user_id, item_id, rating)
        self.mab.update_rewards(self.last_armidx, reward, False)        
        return 


    def epoch_update(self):
        """Defined for compatibility"""
        
        return 


            
