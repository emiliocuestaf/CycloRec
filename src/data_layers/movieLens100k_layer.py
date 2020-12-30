import pandas as pd
import numpy as np
#import random as rd
#from sklearn.neighbors import NearestNeighbors
#from scipy.sparse import csr_matrix

# OWN
from cyclorec.data_layers.data_layer import DataLayer


class MovieLens100kLayer(DataLayer):
    """ DataLayer designed to automatize the work with the ML100k dataset. All ratings are considered the same and a random partition is performed. """

    def __init__(self, name, test_proportion=0.2, partition=1):
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
        users = pd.read_csv("./data/ml-100k/u.user", sep="|", engine='python', names=['user_id', 'gender', 'age', 'occupation', 'zip-code'], header=None) 
        users.index = users['user_id']

        # ITEMS
        # index = id || id |  title | genres
        items = pd.read_csv("./data/ml-100k/u.item",  sep="|", engine='python', usecols=[0,1,2], names=['item_id', 'title', 'release'], header=None)
        items.columns = items.columns.str.lower()
        items.index = items.item_id
        
        if test_proportion == 0.2:
            # TRAIN AND UTILITY MATRIX
            file_prefix = "./data/ml-100k/u{}.".format(partition)
            training_file = file_prefix + 'base'
            training_ratings = pd.read_csv(training_file, sep="\t", engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None)
            training_ratings['rating'] = training_ratings['rating'].replace(rating_conversor)

            # TEST SUBSET AND MATRIX
            test_file = file_prefix + 'test'
            test_ratings = pd.read_csv(test_file, sep="\t", engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None)
            test_ratings['rating'] = test_ratings['rating'].replace(rating_conversor)        

            super().__init__(name, users=users, items=items,  splitted=True, test_proportion=test_proportion, train_set=training_ratings, \
                            test_set=test_ratings, relevance_threshold=relevanceThreshold, antiRelevance_threshold=antiRelevanceThreshold)

        else:
            # TRAIN AND UTILITY MATRIX
            datafile = "./data/ml-100k/u.data"
            all_ratings = pd.read_csv(datafile, sep="\t", engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None)
            all_ratings['rating'] = all_ratings['rating'].replace(rating_conversor)        

            super().__init__(name, users=users, items=items,  splitted=False, whole_set=all_ratings, test_proportion=test_proportion, \
                            relevance_threshold=relevanceThreshold, antiRelevance_threshold=antiRelevanceThreshold)

