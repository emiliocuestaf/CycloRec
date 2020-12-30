import pandas as pd
import numpy as np

# OWN
from cyclorec.data_layers.data_layer import DataLayer

class Cm100kLayer(DataLayer):
    """ DataLayer designed to automatize the work with the Cm100K dataset. All ratings are considered the same and a random partition is performed.
        Every rating, whether the item was known by the user or not, is recommendable.
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
        super().__init__(name, items=items, splitted=False, whole_set=ratings, test_proportion=test_proportion, \
                        relevance_threshold=relevanceThreshold, antiRelevance_threshold=antiRelevanceThreshold )
