import pandas as pd
import numpy as np
import math
import time


class DataLayer():
    """ This class is a generic layer to abstract the functionality of reading/writing from 
        recommendation data. The following attributes are automatically generated from the arguments 
        passed to the constructor.
        Of course, if the data can't be fitted into the suggested format, the programmer can still
        make its own layer overriding any of the given methods.

        These recommendable class attributes are:

        self.items: pandas.DataFrame containing the items with the following format:
                    Index = item_id || item_id | attribute_name1 | attribute_name2 | attribute_name3 ...

        self.users: pandas.DataFrame containing the users with the following format:
                    Index = user_id || user_id | attribute_name1 | attribute_name2 | attribute_name3 ...

        self.initial_training_ratings: pandas.DataFrame containing only the training ratings. Following format:
                                       index = - || user | item | rating
                                       This log is not updated.

        self.utility_matrix: pandas.DataFrame containing only the known ratings with the following format:

                                Index = user_id | item_id1 | item_id2 | item_id3 ...
                                    user_id1    |  r(1,1)  | r(1,2)   | r(1,3)   ...
                                    user_id2    |  r(2,1)  | r(2,2)   | r(3,3)   ...
                                    user_id3    |  r(2,1)  | r(2,2)   | r(3,3)   ...
                                    user_id4    |  r(2,1)  | r(2,2)   | r(3,3)   ...
                                    ...            ...       ...        ...      ...
                            
                             It keeps being updated as long as the recommender system keeps discovering new ratings. 

        self.test_set: pandas.DataFrame containing only the testing ('unknown') ratings with the following format:
                       user_id (index_level_1) | item_id (index_level_2) | rating

        self.relevanceThreshold: Integer or float acting as the threshold over which ratings
                                 are considered 'positive ratings'

        self.relevantCount: Number of positive (relevant), >= relevanceThreshold,  ratings in the test set

        self.antiRelevanceThreshold: integer or float acting as the threshold under which ratings
                                     are considered 'negative ratings'

        self.antiRelevantCount: number of negative (nonRelevant) ratings, <= antiRelevanceThreshold, in the test set

        self.toDiscoverCount: self.antiRelevantCount + self.relevantCount
    """


    def __init__(self, name, users=None, items=None, splitted=False, whole_set=None, test_proportion=1, train_set=None, test_set=None, relevance_threshold=3, antiRelevance_threshold=3):
        """ Constructor. A dataLayer is an structure used to read/write the ratings data independently of the way they are stored in memory.

        Args:
            name (str): Name of the layer

        Keyword Arguments:
            users (Pandas DataFrame):   Contains the users with the following format:
                                        Index = user_id | attribute_name1 | attribute_name2 | attribute_name3 (Column order is not important).
                                        If not passed, a default df containing only the id (extracted from de ratings) will be created.  {Default:None}

            items (Pandas DataFrame):   Contains the items with the following format:
                                        Index = item_id | attribute_name1 | attribute_name2 | attribute_name3 (Column order is not important)
                                        If not passed, a default df containing only the id (extracted from de ratings) will be created.  {Default:None}

            splitted (bool): If True, the user must pass both the train set and the test set as arguments (train_set, test_set).
                             If False, the train and set datasets will be generated from 'whole_set' (uniformtly at random) using the 'test_proportion' argument.  {Default:False}

            whole_set (pandas.DataFrame): pd.Dataframe containing all the ratings. 
                                          FORMAT: user_id  | item_id | rating {Default:None}

            test_proportion (float): Proportion of known ratings that will be stored in the test set. Must be within the [0,1] interval. {Default: 1}

            train_set (pandas.DataFrame): pd.Dataframe containing only the train ratings.
                                          FORMAT: user_id  | item_id | rating {Default:None} 

            test_set (pandas.DataFrame): pd.Dataframe containing only the test ratings. 
                                         FORMAT: user_id  | item_id | rating {Default:None} 

            relevance_threshold (float): Value over (or equal) which ratings are considered relevant {Default: 3}

            antiRelevance_threshold (float): Value under (or equal) which ratings are considered antirelevant (different from non-relevant). Antirelevant may equal to 'disgusting' {Default: 3}
        """

        self.name = name
        self.relevanceThreshold = relevance_threshold
        self.antiRelevanceThreshold = antiRelevance_threshold

        # Ratings
        if splitted == True:
            self.initial_train_set = train_set.copy()
            self.utility_matrix = train_set.pivot_table(index='user_id', columns='item_id', values='rating')

        elif splitted == False:
            # Random sample
            mask = np.random.rand(whole_set.shape[0]) < test_proportion
            # Train
            train_set = whole_set[~mask]
            self.initial_train_set = train_set.copy()
            if self.initial_train_set.empty == True:
                self.utility_matrix = pd.DataFrame()
            else:
                self.utility_matrix = train_set.pivot_table(index='user_id', columns='item_id', values='rating')
            # Test
            test_set = whole_set[mask]
            if test_set.empty == True:
                self.test_df = pd.DataFrame()
            else:
                self.test_df = test_set.copy()
        else:
            raise TypeError('\'splitted\' must be a boolean. It must be either True or False')

        # Users 
        if users is None:
            all_user_ids = self.utility_matrix.index.union(self.test_df['user_id'].unique(), sort=None)
            self.users = pd.DataFrame(index=all_user_ids)
        else:
            if not users.index.is_unique:
                raise TypeError('Users indexes are not unique')

            self.users = users.copy()
            # Adding all the users which don't have any associated rating in the train set
            for user_id in self.users.index:
                if user_id not in self.utility_matrix.index:
                    self.utility_matrix.append(pd.Series(name=user_id))

        # Items
        if items is None:
            all_item_ids = self.utility_matrix.columns.union(self.test_df['item_id'], sort=None)
            self.items = pd.DataFrame(index=all_item_ids)
        else:
            if not items.index.is_unique:
                raise TypeError('Items indexes are not unique')

            self.items = items.copy()
            # Adding all the items which don't have any associated rating in the train set
            for item_id in self.items.index:
                if item_id not in self.utility_matrix.columns:
                    self.utility_matrix[item_id] = np.nan

        # Counting relevant/antirelevant values
        self.relevantCount = 0
        self.antiRelevantCount = 0
        test_value_counts = self.test_df['rating'].value_counts()
        for value in test_value_counts.keys():
            if not np.isnan(value):
                if value <= self.antiRelevanceThreshold:
                    self.antiRelevantCount += test_value_counts[value]
                if value >= self.relevanceThreshold:
                    self.relevantCount += test_value_counts[value]
        self.toDiscoverCount = self.antiRelevantCount + self.relevantCount

        # CHANGING IDS FOR INDEXES
        # Sorting everything 
        # Now, the i-th row of utility_matrix and the i-th row of the users set corresponds to the same user.
        # Also, the i-th column of the utility_matrix corresponds to the i-th row of the items.
        # All the elements from outside the datalayer will work with indexes (much faster)
        self.items = self.items.sort_index(axis=0, ascending=True)
        self.users = self.users.sort_index(axis=0, ascending=True)
        self.utility_matrix = self.utility_matrix.sort_index(axis=0, ascending=True).sort_index(axis=1, ascending=True)
        self.utility_matrix.reset_index(drop=True, inplace=True)
        self.utility_matrix.columns = range(self.items.shape[0])

        # Changing IDS by indexes.
        self.initial_train_set.loc[:, 'user_idx'] = self.initial_train_set.loc[:,'user_id'].apply(lambda x: self.users.index.get_loc(x))
        self.initial_train_set.loc[:, 'item_idx'] = self.initial_train_set.loc[:,'item_id'].apply(lambda x: self.items.index.get_loc(x))
        self.initial_train_set = self.initial_train_set.drop(columns=['user_id', 'item_id'])

        self.test_df.loc[:, 'user_idx'] = self.test_df.loc[:,'user_id'].apply(lambda x: self.users.index.get_loc(x))
        self.test_df.loc[:, 'item_idx'] = self.test_df.loc[:,'item_id'].apply(lambda x: self.items.index.get_loc(x))
        self.test_df = self.test_df.drop(columns=['user_id', 'item_id'])
        self.test_df = self.test_df.set_index(['user_idx', 'item_idx'])

        # Changing indexes both in items and users dfs. The old id is stored as a column
        try:
            self.items = self.items.drop(columns=['item_id'])
        except KeyError:
            pass
        self.items.index.name = 'item_id'
        self.items = self.items.reset_index()
        try:
            self.users = self.users.drop(columns=['user_id'])
        except KeyError:
            pass
        self.users.index.name = 'user_id'
        self.users = self.users.reset_index()

        # Conversion to dict because it seems much faster. This dict will have THE INDEXES AS KEYS
        self.test_set = {level: self.test_df.xs(level).to_dict('index') for level in self.test_df.index.levels[0]}

    def get_name(self):
        """ Returns the name of the datalayer
        Returns:
            str: name
        """
        return self.name

    ###########################################################
    ###                                                     ###
    ###                ITEM OPERATIONS                      ###
    ###                                                     ###
    ###########################################################

    def get_items(self):
        """  Gives all the items in the dataset and its attributes

        Returns:
            pandas.Dataframe -- Pandas dataframe with item ID as index and the rest of attributes as different columns
        """
        return self.items

    def get_item_id_from_index(self, item_idx):
        """ 
        Returns:
        """
        return self.items.iloc[item_idx].loc['item_id']

    def get_n_items(self):
        """ Returns the number of different items in the datalayer

        Returns:
            int: num items
        """
        return self.items.shape[0]

    ###########################################################
    ###                                                     ###
    ###                USER OPERATIONS                      ###
    ###                                                     ###
    ###########################################################

    def list_users(self):
        """  Returns a list containing all the user ids

        Returns:
            list -- List containing all the ids
        """
        return list(self.users.index.values)
    
    def get_users(self):
        """  Gives all the users in the dataset and its attributes

        Returns:
            pandas.Dataframe -- Pandas dataframe with user ID as index and the rest of attributes as different columns
        """
        return self.users

    def get_user_id_from_index(self, user_idx):
        """ 
        Returns:
        """
        return self.users.iloc[user_idx].loc['user_id']

    def get_n_users(self):
        """ Returns the number of different users in the dataframe

        Returns:
            int: num users
        """
        return self.users.shape[0]

    ###########################################################
    ###                                                     ###
    ###             TRAINING SET OPERATIONS                 ###
    ###                  (KNOWN DATA)                       ###
    ###                                                     ###
    ###########################################################

    def get_initial_training_ratings_indexes(self):
        """ Returns all the initial training partition.

        Returns:
            pd.dataframe: dataframe with columns  generic_index || user | item | rating
        """
        return self.initial_train_set[['user_idx', 'item_idx', 'rating']]


    def get_user_initial_recommendations(self, user_idx):
        """ Returns a series which contains the recommendations a user has received in an early stage of the recommendation task
            This is, the recommendations that fall into the train set from a first moment.

        Args:
            user_id ([type]): User index which info is retrieved

        Returns:
            []: Yields a list with the indexes of the items that have been recommended to user_id
        """
        try:
            L = list(self.initial_train_set.loc[self.initial_train_set['user_idx'] == user_idx, 'item_idx'])
            return L
        except KeyError:
            return []

    def get_utility_matrix(self):
        """ Returns the whole utility_matrix

        Returns:
            pandas.DataFrame -- Utility matrix.
                                Rows = users_ids
                                Columns = items_ids
                                Content = corresponding ratings
        """
        return self.utility_matrix


    def get_utility_submatrix(self, users_idxs=None, items_idxs=None):
        """ Returns a subset of the utility matrix in which rows = [user_ids] and columns = [item_ids]

        Arguments:
            users_ids {list} -- Users_idxs to perform the row-filtering. If None all the users are taken {Default:None}
            items_ids {list} -- Items_idxs to perform the column-filtering. If None all the items are taken {Default:None}

        Returns:
            pandas.DataFrame -- Submatrix of the utility matrix.
                                Rows = users_ids
                                Columns = items_ids
                                Content = corresponding rewards
        """

        if  (users_idxs is None) and  (items_idxs is None):
            return self.utility_matrix

        elif users_idxs is None:
            return self.utility_matrix.iloc[:, items_idxs]

        elif items_idxs is None:
            return self.utility_matrix.iloc[users_idxs, :]

        return self.utility_matrix.iloc[users_idxs, items_idxs]


    def get_known_rating(self, user_idx, item_idx):
        """ Get the rating for (user, item) from the utility matrix

        Arguments:
            user_id {list} -- user index
            item_id {list} -- item index

        Returns:
            int -- rating
        """
        return self.utility_matrix.iat[user_idx, item_idx]


    def get_user_known_ratings(self, user_idx):
        """ Returns all the ratings given by a user in the training set

        Arguments:
            user_id {int} -- user_id

        Returns:
            pandas.Series --  Index equals the item indexes and value equals the rating
        """
        return self.utility_matrix.iloc[user_idx].dropna().rename('rating')


    def set_rating(self, user_idx, item_idx, rating):
        """ Sets a reward for the tuple (user_id, item_id) in the utility matrix
            This method is used to keep the training set updated while evaluating over the
            test set.

        Arguments:
            user_id {int} -- user index
            item_id {int} -- item index
            rating  {int} -- rating
        """
        #In the case the user_id or the item_id does not have any recommendation yet, 'at' crates automatically its corresponing row/column.
        if math.isnan(rating):
            return 
        else:
            self.utility_matrix.at[user_idx, item_idx] = rating
            return

    ###########################################################
    ###                                                     ###
    ###           TEST SET AND REWARD OPERATIONS            ###
    ###                                                     ###
    ###########################################################


    def get_reward(self, user_id, item_id, rating):
        """ Returns a reward specially customized for bandits (as they might not be working in the same scale as tthe ratings)
            By default, it will return the proper rating. This is a method designed for overriding if it was necessary.

        Args:
            user_id {int} -- user index
            item_id {int} -- item index
            rating (pd.Series): Series that has at least the attribute 'rating' which contains the value stored in the test_set or np.nan.
                                It can also contain more attributes.

        Returns:
            float : the reward that will be passed to the bandit
        """
        return rating['rating']

    def get_user_test_ratings_dict(self, user_idx):
        if user_idx in self.test_set.keys():
            return self.test_set[user_idx]
        else:
            return {}
    ###########################################################
    ###                                                     ###
    ###                METRICS OPERATIONS                   ###
    ###                                                     ###
    ###########################################################

    def isRelevant(self, rating):
        """ A function to check whether a rating is relevant or not, which result depends on the threshold defined as a class variable.

        Args:
            rating (numeric): Rating to classify

        Returns:
            bool: True if the rating is relevant and False if it is not. All "nan" ratings are NOT RELEVANT.
        """
        rating = rating['rating']
        if math.isnan(rating):
            return False
        return rating > self.relevanceThreshold

    def isAntiRelevant(self, rating):
        """ A function to check whether a rating is antiRelevant or not, which result depends on the threshold defined as a class variable.

        Args:
            rating (numeric): Rating to classify

        Returns:
            bool: True if the rating is AntiRelevant and False if it is not. All "nan" ratings are NOT ANTIRELEVANT.
        """
        rating = rating['rating']
        if math.isnan(rating):
            return False
        return rating < self.antiRelevanceThreshold


    def get_relevant_count(self):
        """ Returns the number of relevant ratings in the test subset (which have to be discovered)
            Relevant ratings are those stricly higher than the instance variable self.relevanceThreshold (should be defined at the constructor)

        Returns:
            int -- number of relevant ratings
        """
        return self.relevantCount


    def get_antirelevant_count(self):
        """ Returns the number of antirelevant ratings in the test subset.
            AntiRelevant ratings are those strictly lower than the instance variable self.antiRelevanceThreshold (should be defined in the constructor)


        Returns:
            int -- number of antirelevant ratings
        """
        return self.antiRelevantCount


    def get_toDiscover_count(self):
        """ Returns the number of ratings that have any kind of useful information in the test set, they are those which are relevant plus those which are antiRelevant.

        Returns:
            int -- number of relevant and antirelevant ratings
        """
        return self.toDiscoverCount
