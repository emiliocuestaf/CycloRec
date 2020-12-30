import pandas as pd
import numpy as np
# TEST
import time
# OWN
from cyclorec.recommenders.base_recommender import BaseRecommender
from cyclorec.custom_exceptions.custom_exceptions import UserFullError, NoPositiveFeedbackError

TIME = True

class RandomRecommender(BaseRecommender):
    """ Recommender that always recommends a random item. Inherits form BaseRecommender (abstract class)  """


    def __init__(self, dataLayer):
        super().__init__('random', dataLayer)
    

    ###########################################################
    ###                                                     ###
    ###                RECOMMEND FUNCTION                   ###
    ###                                                     ###
    ###########################################################


    def select_item(self, userLayer, enable_repetitions=False):
        """Recommends an item randomly to the selected user
        
        Arguments:
            userLayer {UserLayer} -- userLayer of the user who will receive the recommendation
        
        Keyword Arguments:
            enable_repetitions {bool} -- True if repeating items is allowed, False if not (default: {False})
        
        Returns:
            pandas.Series -- contains the recommended item
        """
        eligible_items_ids = userLayer.get_user_available_items_idxs(enable_repetitions)

        if len(eligible_items_ids)==0:
            raise UserFullError(userLayer.get_user())
        else:
            item_idx = np.random.choice(eligible_items_ids)
            return item_idx
        



    ###########################################################
    ###                                                     ###
    ###               RECOMMENDER UPDATINGS                 ###
    ###                                                     ###
    ###########################################################
    def train(self):
        """ Method to implement the default training behaviour. It will only be executed once at the beginning of the recommendation task."""
        return  
    
    def rating_update(self, user_id, item_id, reward):
        """ Defined for compatibility with BaseRecommender  """
        return 


    def epoch_update(self):
        """ Defined for compatibility with BaseRecommender  """
        return




