import pandas as pd
import numpy as np
import random
# TEST
import time
# OWN
from cyclorec.recommenders.base_recommender import BaseRecommender
from cyclorec.custom_exceptions.custom_exceptions import UserFullError, NoPositiveFeedbackError


class MostValuableRecommender(BaseRecommender):
    """ Recommender that always recommends the item with the best mean valoration. Inherits form BaseRecommender (abstract class)  """

    def __init__(self, dataLayer):
        super().__init__('most_valuable', dataLayer)
        self.most_popular_counts = np.array([])
        self.smoothing_alpha = 1
        self.smoothing_delta = 2
        self.smoothing_div = self.smoothing_alpha*self.smoothing_delta
        self.counts = np.array(self.smoothing_div)
        # laplacian smoothing
        # init = alpha/(alpha*delta) = 0.5
       

    ###########################################################
    ###                                                     ###
    ###                RECOMMEND FUNCTION                   ###
    ###                                                     ###
    ###########################################################


    def select_item(self, userLayer, enable_repetitions=False):
        """Recommends an item by valorations (item with the best mean valoration) to the selected user 
        
        Arguments:
            userLayer {UserLayer} -- userLayer of the user who will receive the recommendation
                    
        Keyword Arguments:
            enable_repetitions {bool} -- True if repeating items is allowed, False if not (default: {False})
        
        Returns:
            pandas.Series -- contains the recommended item
        """

        eligible_items_idxs = userLayer.get_user_available_items_idxs()
        if len(eligible_items_idxs)==0:
            raise UserFullError(userLayer.get_user())
        
        means = self.most_valuable_means[eligible_items_idxs]
        unties = eligible_items_idxs[means == np.max(means)]
        if unties.shape[0] != 1:
            most_valuable_item_id = np.random.choice(unties)
        else:
            most_valuable_item_id = unties[0]
       

        return most_valuable_item_id


    ###########################################################
    ###                                                     ###
    ###               RECOMMENDER UPDATINGS                 ###
    ###                                                     ###
    ###########################################################
    def train(self):
        """ Method to implement the default training behaviour. It will only be executed once at the beginning of the recommendation task."""
        
        sums = np.array(self.data_layer.get_utility_matrix().sum().fillna(0).values)
        self.counts = np.array(self.data_layer.get_utility_matrix().count().values + self.smoothing_div)
        self.most_valuable_means = (sums + self.smoothing_alpha)/self.counts
        return  

    def rating_update(self, user_id, item_idx, reward):
        """ Must be defined for compatibility with abstract class BaseRecommender  """
        reward = reward['rating']
        if reward is not np.nan and reward == 1:
            old_count = self.counts[item_idx]
            self.counts[item_idx] += 1 
            self.most_valuable_means[item_idx] = (self.most_valuable_means[item_idx]*old_count + reward)/self.counts[item_idx]
        else:
            old_count = self.counts[item_idx]
            self.counts[item_idx] += 1 
            self.most_valuable_means[item_idx] = (self.most_valuable_means[item_idx]*old_count)/self.counts[item_idx]

    def epoch_update(self):
        """ Recalculates the items mean reating"""
        return




