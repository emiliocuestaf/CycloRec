import pandas as pd
import numpy as np
#import random as rd
#from sklearn.neighbors import NearestNeighbors
#from scipy.sparse import csr_matrix

# OWN
from cyclorec.data_layers.data_layer import DataLayer

class FFeedbackCm100kLayer(DataLayer):
    """ The only difference between this layer and Cm100kLayer is the feedback the algorihtms are receiving. 
        In the typical version, if an algorithm succeded in a recommendation it would be updated with a reward.
        However, in FFeedback the algorithm will only be rewarded if the recommended item was known by the user.  
    """
    
    def __init__(self, name, test_proportion):
        """ Constructor

        Args:
            name (str): DataLayer name
            test_proportion (float): proportion of the whole set to be assigned to the test set
        """
        # CUSTOMIZABLE RATING VARIABLES
        rating_normalization = 'bin'
        if rating_normalization == 'bin':
            rating_conversor = dict({1: 0, 2: 0, 3: 1, 4: 1})
            relevanceThreshold = 0.5
            antiRelevanceThreshold = 0.5
        else:
            rating_conversor = dict({1: -1, 2: 0, 3: 1, 4: 1})
            relevanceThreshold = 0.5
            antiRelevanceThreshold = -0.5

        # ITEMS
        # index = id || item_id | url | title | artist
        items = pd.read_csv("./data/cm100k/items.txt", sep="\t")
        items.columns = items.columns.str.lower()
        items = items.rename(columns={"item": "item_id"})
        items.index = items.item_id


        # ALL RATINGS
        ### Dataframe with ratings
        # index || user | item | rating | known
        ratings = pd.read_csv("./data/cm100k/ratings.txt", sep="\t", names=['user_id', 'item_id', 'rating', 'known'], header=None)
        ratings.columns = ratings.columns.str.lower()
        ratings['rating'] = ratings['rating'].replace(rating_conversor)
        super().__init__(name, items=items,  splitted=False, whole_set=ratings, test_proportion=test_proportion, \
                        relevance_threshold=relevanceThreshold, antiRelevance_threshold=antiRelevanceThreshold )


    #override
    def get_bandit_reward(self, user_id, item_id, rating):
       
        if np.isnan(rating['rating']):
            return np.nan
        elif rating['known'] == 1:
            return rating['rating']
        else:
            return np.nan

            