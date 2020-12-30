import pandas as pd
import numpy as np
#import random as rd
#from sklearn.neighbors import NearestNeighbors
#from scipy.sparse import csr_matrix

# OWN
from cyclorec.data_layers.data_layer import DataLayer


class MovieLens1MLayer(DataLayer):
    """ DataLayer designed to automatize the work with the ML1M dataset. All ratings are considered the same and a random partition is performed. """

    def __init__(self, name, test_proportion):
        """ Constructor

        Args:
            name (str): DataLayer name
            test_proportion (float): proportion of the whole set to be assigned to the test set
        """
        # CUSTOMIZABLE RATING VARIABLES 
        rating_normalization = 'bin'
        if rating_normalization == 'bin':
            rating_conversor = dict({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})
            relevanceThreshold = 0.5
            antiRelevanceThreshold = 0.5
        else: 
            rating_conversor = dict({1: -1, 2: -1, 3: 0, 4: 1, 5: 1})
            relevanceThreshold = 0.5
            antiRelevanceThreshold = -0.5

        # USERS
        users = pd.read_csv("./data/ml-1m/users.dat", sep="::", engine='python', names=['user_id', 'gender', 'age', 'occupation', 'zip-code'], header=None) 
        users.index = users['user_id']

        # ITEMS
        # index = id || id |  title | genres
        items = pd.read_csv("./data/ml-1m/movies.dat",  sep="::", engine='python', names=['item_id', 'title', 'genres'], header=None)
        items.columns = items.columns.str.lower()
        items.index = items.item_id
        
        # RATINGS
        # index || user | item | rating | known
        ratings = pd.read_csv("./data/ml-1m/ratings.dat", sep="::", engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None)
        ratings.columns = ratings.columns.str.lower()
        ratings['rating'] = ratings['rating'].replace(rating_conversor)
        
        super().__init__(name, users=users, items=items, splitted=False, whole_set=ratings, test_proportion=test_proportion, \
                        relevance_threshold=relevanceThreshold, antiRelevance_threshold=antiRelevanceThreshold)




