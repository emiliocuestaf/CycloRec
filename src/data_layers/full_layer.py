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
        users = range(1,51)
        items = range(1,51)

        ratings = pd.DataFrame()

        for user in users:
            cpit = [item for item in items]
            while len(cpit)!=0:
                ind = np.random.randint(0, len(cpit))
                item = cpit.pop(ind)
                rating = np.random.randint(0, 2)
                ratings = ratings.append(pd.Series([user, item, rating]), ignore_index=True)

        ratings.columns = ['user_id', 'item_id', 'rating']

        super().__init__(name, items=items, splitted=False, whole_set=ratings, test_proportion=test_proportion, \
                        relevance_threshold=0.5, antiRelevance_threshold=0.5 )
