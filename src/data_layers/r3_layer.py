import pandas as pd
import numpy as np

# OWN
from cyclorec.data_layers.data_layer import DataLayer


class R3Layer(DataLayer):
    """ DataLayer designed to automatize the work with the R3 dataset. All ratings are considered the same and a random partition is performed. """

    def __init__(self, name, test_proportion):
        """ Constructor

        Args:
            name (str): DataLayer name
            test_proportion (float): proportion of the whole set to be assigned to the test set
        """
        # CUSTOMIZABLE RATING VARIABLES 
        rating_normalization = 'bin'
        if rating_normalization == 'bin':
            rating_conversor = dict({1: 0, 2: 0, 3: 1, 4: 1, 5: 1})
            relevanceThreshold = 0.5
            antiRelevanceThreshold = 0.5
        else: 
            rating_conversor = dict({1: -1, 2: -1, 3: 0, 4:1, 5: 1})
            relevanceThreshold = 0.5
            antiRelevanceThreshold = -0.5


        # TRAIN SET 
        # ['user_id', 'item_id', 'rating']
        training_ratings = pd.read_csv("./data/R3/ydata-ymusic-rating-study-v1_0-train.txt",  sep="\t", engine='python', names=['user_id', 'item_id', 'rating'], header=None)      
        training_ratings['rating'] = training_ratings['rating'].replace(rating_conversor)
  
        # TEST  SET
        # ['user_id', 'item_id', 'rating']
        test_ratings = pd.read_csv("./data/R3/ydata-ymusic-rating-study-v1_0-test.txt", sep="\t", engine='python', names=['user_id', 'item_id', 'rating'], header=None)   
        test_ratings['rating'] = test_ratings['rating'].replace(rating_conversor)

        all_ratings = pd.concat([test_ratings, training_ratings])
        real_all_ratings = all_ratings[all_ratings['user'].isin(test_ratings['user'])]

        # We wont use the default train set, we will split it automatically

        super().__init__(name, users=None, items=None,  splitted=False, whole_set=real_all_ratings, test_proportion=test_proportion, \
                        relevance_threshold=relevanceThreshold, antiRelevance_threshold=antiRelevanceThreshold)
        


