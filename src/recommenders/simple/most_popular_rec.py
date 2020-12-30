import pandas as pd
import numpy as np
# TEST
import time
# OWN
from cyclorec.recommenders.base_recommender import BaseRecommender
from cyclorec.custom_exceptions.custom_exceptions import UserFullError, NoPositiveFeedbackError


class MostPopularRecommender(BaseRecommender):
    """ Recommender that always recommends the most popular (the one with more ratings) item
    """

    def __init__(self, dataLayer):
        super().__init__('most_popular', dataLayer)
        self.most_popular_counts = np.array([])

    ###########################################################
    ###                                                     ###
    ###                RECOMMEND FUNCTION                   ###
    ###                                                     ###
    ###########################################################


    def select_item(self, userLayer, enable_repetitions=False):
        """Recommends an item by popularity (chosing the item with more valorations)
        
        Arguments:
            userLayer {UserLayer} -- userLayer of the user who will receive the recommendation
                    
        Keyword Arguments:
            enable_repetitions {bool} -- True if repeating items is allowed, False if not (default: {False})
        
        Returns:
            pandas.Series -- contains the recommended item
        """
                
        eligible_items_idxs = userLayer.get_user_available_items_idxs(enable_repetitions)
        if len(eligible_items_idxs) == 0:
            raise UserFullError(userLayer.get_user())

        counts = self.most_popular_counts[eligible_items_idxs]      
        unties = eligible_items_idxs[counts == np.max(counts)]

        if unties.shape[0] != 1:
            most_popular_itemid = np.random.choice(unties)
        else:
            most_popular_itemid = unties[0]
        
        return most_popular_itemid


    ###########################################################
    ###                                                     ###
    ###               RECOMMENDER UPDATINGS                 ###
    ###                                                     ###
    ###########################################################

    def train(self):
        """ Method to implement the default training behaviour. It will only be executed once at the beginning of the recommendation task."""
        self.most_popular_counts = np.array(self.data_layer.get_utility_matrix().count().values)
        return  

    def rating_update(self, user_id, item_id, reward):
        """ Must be defined for compatibility with abstract class BaseRecommender """
        reward = reward['rating']
        if reward is not np.nan and reward == 1:
            self.most_popular_counts[item_id] += 1
        return 

    def epoch_update(self):
        """ Recalculates the ratings-per-item count"""
        #self.most_popular_counts = self.data_layer.get_utility_matrix().count()
        return





