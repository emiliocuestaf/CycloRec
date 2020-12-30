# GENERAL
import pandas as pd
import numpy as np
# OWN
from  cyclorec.data_layers.data_layer import DataLayer

class UserLayer():
    """ Class to run over a DataLayer. Stores information about a single user and the recommendations that have already been made to her/him."""

    def __init__(self, user_idx, data_layer):
        """ Constructor.

        Args:
            user_id (int): User_id, id of the characteristic user.
            data_layer (DataLayer): Subjacent DataLayer
        """
        self.user_idx = user_idx
        self.dataLayer = data_layer
        self.user_id = self.dataLayer.get_user_id_from_index(self.user_idx)

        recommended_items_list = self.dataLayer.get_user_initial_recommendations(user_idx=self.user_idx)
        all_items = self.dataLayer.get_items()
        self.all_items_ids = np.arange(all_items.shape[0])

        available_items = all_items[~all_items.index.isin(recommended_items_list)]
        self.available_items = np.array(available_items.index)
        
        # Test
        self.user_ratings = self.dataLayer.get_user_test_ratings_dict(self.user_idx)
        self.user_rated_items = set(self.user_ratings.keys())

    def get_user(self):
        """ Returns the userLayer's characteristic user
        
        Returns:
            int -- user_id
        """
        return self.user_idx

    ###########################################################
    ###                                                     ###
    ###                   ELIGIBLE ITEMS                    ###
    ###                                                     ###
    ###########################################################

    def get_user_available_items(self, enable_repetitions=False):
        """ Returns the user eligible items. These can be all of the items or only the ones that havent 
            been recommened yet.

        Arguments:
            enable_repetitions {bool} -- If True, all the items are eligible and therefore it returns all the items.
                                            If False, only the not-recommended items are returned  {Default: False}
        
        Returns:
            pandas.Dataframe -- Pandas dataframe of items with itemid as index and the rest of attributes as columns
        """

        if not enable_repetitions:
            return self.dataLayer.get_items().iloc[self.available_items]
        else:
            return self.dataLayer.get_items()

    def get_user_available_items_idxs(self, enable_repetitions=False):
        if enable_repetitions:
            return self.all_items_ids
        else:
            return self.available_items

    def add_recommended_item(self, item_idx):
        """ Removes an item from the available items list. So, it won't be recommended again if repetitions are not allowed
        
        Arguments:
            item_id {int} -- Item to be added
        """

        self.available_items = np.delete(self.available_items, np.where(self.available_items == item_idx))
        return 

    ###########################################################
    ###                                                     ###
    ###               TRAIN SET OPERATIONS                  ###
    ###                                                     ###
    ###########################################################
    
    def get_user_train_ratings(self):
        """ Returns al the user ratings in the train set

        Returns:
            pandas.Series --  Index equals the item id and value equals the rating
            
        """
        return self.dataLayer.get_user_known_ratings(self.user_idx)


    def get_user_train_relevant_ratings(self):
        """ Returns the user relevant ratings in the train set, according to the threshold defined in the dataLayer
            Relevant ratings are those over the relevance threshold

        Returns:
            pandas.Series --  Index equals the item id and value equals the rating

        """
        return self.dataLayer.get_get_user_train_relevant_ratings(self.user_idx)


    def get_user_train_antirelevant_ratings(self):
        """ Returns the user antirelevant ratings in the train set, according to the threshold defined in the dataLayer
            Antirelevant ratings are those under the antirelevance threshold

        Returns:
            pandas.Series --  Index equals the item id and value equals the rating

        """
        return self.dataLayer.get_get_user_train_antirelevant_ratings(self.user)

    ###########################################################
    ###                                                     ###
    ###                TEST SET OPERATIONS                  ###
    ###                                                     ###
    ###########################################################

    def check_rating(self, item_id):
        """ Returns the rating the user 'gives' to the recommended item from the knowledge in the test set.
        
        Arguments:
            item_id {[type]} -- Item to receive the rating
        
        Returns: 
            int -- rating
        """
        return self.dataLayer.get_test_rating(self.user, item_id)

    def get_test_rating(self, item_id):
        if item_id in self.user_rated_items:
            return self.user_ratings[item_id]
        else:
            return {'rating': np.nan}
