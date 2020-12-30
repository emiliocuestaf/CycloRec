# GENERAL
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import random
import math
import time

# DATA LAYERS
from cyclorec.data_layers.data_layer import DataLayer
# USER LAYER
from cyclorec.user_layer.user_layer import UserLayer
# EVALUATION
from cyclorec.evaluator.evaluator import Evaluator
# EXCEPTIONS
from cyclorec.custom_exceptions.custom_exceptions import UserFullError, SystemFullError,  NoPositiveFeedbackError


class BaseRecommender(ABC):
    """ BaseRecommender abstract class, which simulates the basic funcionalities of a recommender system. 
        All the code which concerns choosing users, looking for rewards, interactive cycles... is implemented here.
        Every other recommender inherits from this class and must implement just the abstract methods shown 
        after the constructor. The non-implemented are basically the way to choose an item, the way an algorithm
        improves by using the train set and the way to self-update when discovering relevant and antirelevant items"""
        
    def __init__(self, recommenderName, dataLayer):        
        """ Constructor.

        Args:
            recommenderName (str): Name of the recommender. For example, 'Gladys'
            dataLayer (DataLayer): DataLayer from which the system will take the items, the users and the ratings. 
        """
        
        self.recommender_name = recommenderName 
        self.data_layer = dataLayer
        # Data structure to handle user profiles
        self.user_layers = {}
        # Instance to evaluate
        self.evaluator = Evaluator(self.data_layer)
        # Loading the list of users which must be recommended in a round (round = single recommendation for each of the users in the system)
        self.target_users_list = self.data_layer.list_users()
        self.target_users_set = set(self.target_users_list)

        super().__init__()
        
    
    ###########################################################################
    ###                                                                     ###
    ###                        SPECIFIC FUNCTIONALITY                       ###
    ###       (Every subclass (recommender) must implement these methods)   ###
    ###                                                                     ###
    ###########################################################################


    @abstractmethod
    def select_item(self, userLayer, enable_repetitions=False):
        """ Chooses an available item for a concrete user.
        
        Arguments:
            userLayer {UserLayer} -- layer attached to the user who is going to receive the recommendation.
            enable_repetitions {bool} -- True if repeating items is allowed, False if not (default: {False}).
        
        Returns:
            pandas.Series -- contains the recommended item
        """
        pass

    @abstractmethod
    def train(self):
        """ Method to implement the default training behaviour. It should only be executed once at the beginning of the recommendation task.
        """
        pass
    
    @abstractmethod
    def rating_update(self, user, item, rating):
        """ Update to be made right after every single recommendation
        
        Arguments:
            user {int}   -- User from last recommendation
            item {int}   -- Recommended item
            rating {int} -- Reward from last recommendation
        """
        pass
    
    @abstractmethod
    def epoch_update(self):
        """ Update to be performed only during an epoch change. 
            This function was thought for models with big updating times and therefore updating them at each iterarion is not afordable.   
        """
        pass 


    ###################################################################################################################################
    ###                                                                                                                             ###
    ###                                             BASE_RECOMMENDER FUNCTIONALITY                                                  ###
    ###                       To implement your custom recommender reading below this point is not necessary                        ###   
    ###                                                                                                                             ###
    ###################################################################################################################################

    ###########################################################
    ###                                                     ###
    ###                BASIC OPERATIONS                     ###
    ###          Recommendation and evaluation              ###
    ###                                                     ###
    ###########################################################


    def recommend(self, user_id, enable_repetitions=False):
        """ Recommends a single item to a concrete user. Stores the item as "already recommended" for each user.
            If the user cannot be recommended more items, removes it from the targte users list.
        
        Arguments:
            user_id {int} -- User to be recommended
        
        Keyword Arguments:
            enable_repetitions {bool} -- True: Same item can be recommended twice or more times 
                                         False: Each item can only be recommended once to each user (default: {False})
        
        Raises:
            UserFullError: Raises when the user to be recommended can't be recommended more items (When enable_repetitions == False and there are no more items to recommend)
        
        Returns:
            item -- The recommended item (As a pandas series with all its attributes, if they were available)
        """

        # In the first appearance of a user, its user layer is created from scratch
        if user_id not in self.user_layers.keys():
            self.user_layers[user_id] = UserLayer(user_id, self.data_layer)
        

        # Selecting layer
        userLayer = self.user_layers[user_id]

        # RECOMMENDATION
        item = self.select_item(userLayer, enable_repetitions)
      
        # We add the recommended item to the user's vetoed list in order to avoid it being recommended again if enable_repetitions == False
        #userLayer.add_recommended_item(item['item_id'])
        userLayer.add_recommended_item(item)

        return item

    def check_rating(self, user_id, item_id):
        """ Looks for the rating associated to the (user_id, item_id) tuple in the test set (see dataLayer). 
            It also updates the train set with the acquired information.
        
        Arguments:
            user_id {int} -- User who has been recommended
            item_id {int} -- Item which has been recommended
        
        Returns:
            rating {pd.Series} -- Contains the associated rating in the 'rating' field. Also returns any other
                                  field present in the test_set from the dataLayer.
        """
        # 1. Checking the test set valoration for this (user, item) tuple.
        rating = self.user_layers[user_id].get_test_rating(item_id)
        #rating = self.data_layer.get_test_rating(user_id, item_id)

        # 2. Updates the train utility matrix
        value = rating['rating']
        self.data_layer.set_rating(user_id, item_id, value)        
        return rating
    

    def eval_recommendation(self, rating):
        """ Calculates the results of different metrics using the test set rating obtained (usually) by check_reward. 
            At the present time, it returns the following metrics: 'recall', 'precision', 'fallout', 'antiprecision', 'discovery_rate'
        
        Arguments:
            rating {pd.Series} -- Rating the user "user_id" gave to the item "item_id"
        
        Returns:
            recall, precision, fallout, antiprecision, discovery_rate
        """
        recall, precision, fallout, antiprecision, discovering = self.evaluator.all_metrics(rating)
        return recall, precision, fallout, antiprecision, discovering


    ###########################################################
    ###                                                     ###
    ###           ROUND AND EPOCH FUNCTIONS                 ###
    ###            Simulate real behaviour                  ###
    ###                                                     ###
    ###########################################################

    def recommendation_round(self, enable_repetitions=False, enable_metrics=False):
        """ Makes a single recommendation for each target user and evaluates it in the same step. 
            Keeps the utility matrix updated every time. It also enables the recommender's learning 
            by calling the self.rating_update(...) function after every single recommendation.
        
        Keyword Arguments:
            enable_repetitions {bool} -- If True items that have been recommended before can be recommended again 
                                         Therefore, if True every user is guaranteed to have one recommendation.
                                         If False, items can't be recommended twice to the same user. (default: {False})
            enable_metrics {bool}     -- If True the cumulative metrics (recall, precision, fallout, precision and discovery rate) 
                                         will be computed and returned. (default: {False})

        Raises:
            SystemFullError: Launched when there are no more users to be recommended

        Returns:
            pandas.DataFrame -- a Pandas DataFrame containing the recommendations. Columns = ['user_id', 'item_id', 'received_reward']
            float -- (Only if enable_metrics == True) cumulative recall metric of the whole round based on the training set
            float -- (Only if enable_metrics == True) mean precision metric of the whole round based on the training set
            float -- (Only if enable_metrics == True) cumulative fallout metric of the whole round based on the training set
            float -- (Only if enable_metrics == True) mean antiprecision metric of the whole round based on the training set
            float -- (Only if enable_metrics == True) cumulative discovering of the whole round based on the training set
        """
        # Copy users
        remaining_users = self.target_users_list.copy()
        
        if len(remaining_users) == 0:
            # No round can be made if there are not available users
            raise SystemFullError

        Lrec = []
        # USERS LOOP
        while len(remaining_users) != 0:
            # RANDOM USER SELECTION (Between availables)
            user_id = random.choice(tuple(remaining_users))

            # USER REMOVAL (only 1 rec per user&round is allowed)
            remaining_users.remove(user_id)

            # RECOMMENDATION AND EVALUATION
            try:
                # Previous version returned all the item now we return only the id (much faster)
                # Recommending
                # item = self.recommend(user_id, enable_repetitions)
               
                # # Updating
                # rating = self.check_rating(user_id, item['item_id'])
                # Lrec.append([user_id, item['item_id'], rating])
                # self.rating_update(user_id, item['item_id'], rating)
                
                # Recommending
                item_id = self.recommend(user_id, enable_repetitions)

                # Updating
                rating = self.check_rating(user_id, item_id)

                Lrec.append([user_id, item_id, rating])
                self.rating_update(user_id, item_id, rating)

                # Metrics (they are cumulative)
                if enable_metrics:
                    recall, precision, fallout, antiprecision, discovering = self.eval_recommendation(rating)   

            except UserFullError:
                # If the user cant be recommended more items, it is removed from the available users, the 
                # reccommendation round must continue with the rest of users
                self.target_users_list.remove(user_id)   
                self.target_users_set.remove(user_id)   
                if len(remaining_users) == 0 and len(Lrec) == 0:
                    if enable_metrics:
                        recall, precision, fallout, antiprecision, discovering = self.evaluator.get_metrics()
                    
        # Results dataframe
        recsDF = pd.DataFrame(Lrec, columns=['user_id', 'item_id', 'received_reward'])
        if enable_metrics:
            return recsDF, recall, precision, fallout, antiprecision, discovering
        else:
            return recsDF, -1, -1, -1, -1, -1

    ###########################################################
    ###                                                     ###
    ###             ROUND LOOPS FUNCTIONS                   ###
    ###            Simulate real behaviour                  ###
    ###                                                     ###
    ###########################################################

    def recommendation_round_loop(self, maxiter, enable_repetitions=False, enable_metrics=False, verbose=False):
        """ Simulates and evaluates, one by one, N rounds of recommendations. The recommender can learn from each recommendation due to the
            recommendation_round() function which calls rating_update(). Every epoch, tne function epoch_update() is also called, to allow
            updates that aren't fast enough to be performed just after every recommendation. 
        
        Arguments:
            maxiter {int} -- Number of rounds to simulate. If all the users are full before reaching the maxiter rounds, 
                             the loop stops.
        
        Keyword Arguments:
            enable_repetitions {bool} -- True: Same item can be recommended twice or more times 
                                         False: Each item can only be recommended once to each user (default: {False})
            enable_metrics {bool}     -- If True the cumulative metrics (recall, precision, fallout, precision and discovery rate) 
                                         will be computed and returned. (default: {False})                             
            verbose {bool} -- If True: Terminal will show messages when the whole recommendation process is 20%, 40%, 60%, 80% and 100% fulfilled.
                              If False: Anything will be shown (default: {False})

        
        Returns: 
            [0] pd.DataFrame with the columns=['t', 'user_id', 'item_id', 'received_reward']
            [1] (if enable_metrics == True) pd.DataFrame with the columns=['t', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering']
        """

        recommendationsDF = pd.DataFrame(columns=['t', 'user_id', 'item_id', 'received_reward']) 
        Lmetrics = []

        i = 0
        while i < maxiter:
            # if verbose and math.floor((i+1)/maxiter*100) % 20 == 0 and math.floor((i+2)/maxiter*100) % 20 != 0:
            #     print('\t{}%'.format(math.floor((i+1)/maxiter*100)))

            try: 
                epochDF, recall, precision, fallout, antiprecision, discovering = self.recommendation_round(enable_repetitions=enable_repetitions, enable_metrics=enable_metrics)
               
            except SystemFullError as e:
                # If the system cannot do more recommendations it makes no sense to continue iterating
                # Results are saved and the loop ends
                e.error_msg()
                metricsDF = pd.DataFrame(Lmetrics, columns=['t', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering'])
                return recommendationsDF, metricsDF
            
            if enable_metrics:
                # Storing the metrics corresponding to this iteration
                Lmetrics.append([i+1, recall, precision, fallout, antiprecision, discovering])
                
            # Storing all the recommendations made in this iteration (with epoch=i)
            epochDF['t'] = i+1
            recommendationsDF = recommendationsDF.append(epochDF, ignore_index=True)
            
            # Updating recommender info for the next rec round
            self.epoch_update()
            i += 1
            

        # METRICS
        metricsDF = pd.DataFrame(Lmetrics, columns=['t', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering'])
        return recommendationsDF, metricsDF

    ###########################################################
    ###                                                     ###
    ###                   FIXED USER                        ###
    ###               Useful for testing                    ###
    ###                                                     ###
    ###########################################################

    def fixed_user_recommendation_loop(self, user_id, maxiter, enable_repetitions=False, enable_metrics=False, verbose=False):
        """ Recommends and evaluates, one by one, N recommendations to the specified user.
            If
                1) N is bigger to the number of the items the user has NOT been recommended yet (R)
                AND
                2) enable_repetitions == True, 
            then the loop will stop at R (< maxiter) iterations. 
            It DOES learn in the process.

        
        Arguments:
            user_id {int} -- User to be recommended
            maxiter {int} -- Number of maximum recommendations
        
        Keyword Arguments:
            enable_repetitions {bool} -- True: Same item can be recommended twice or more times 
                                         False: Each item can only be recommended once to each user (default: {False})
            enable_metrics {bool}     -- If True the cumulative metrics (recall, precision, fallout, precision and discovery rate) 
                                         will be computed and returned. (default: {False})        
            verbose {bool} -- If True: Terminal will show messages when the whole recommendation process is 20%, 40%, 60%, 80% and 100% fulfilled.
                              If False: Anything will be shown (default: {False})

        
        Returns: pd.DataFrame with the columns=['user_id', 'item_id', 'received_reward', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering']
        """

        resultsL = []

        i = 1
        while i < maxiter:
        
            if verbose and math.floor((i+1)/maxiter*100) % 20 == 0 and math.floor((i+2)/maxiter*100) % 20 != 0:
                print('\t{}%'.format(math.floor((i+1)/maxiter*100)))

            try: 
                # Recommending
                item_id = self.recommend(user_id, enable_repetitions)

                # Updating
                rating = self.check_rating(user_id, item_id)                
                self.rating_update(user_id, item_id, rating)

                # Metrics (they are cumulative)
                if enable_metrics:
                    recall, precision, fallout, antiprecision, discovering = self.eval_recommendation(rating)    
                    resultsL.append([user_id,  item_id, rating['rating'], recall,  precision,  fallout,  antiprecision, discovering])
                else:
                    resultsL.append([user_id,  item_id, rating['rating']])

                # For fixed user recommendation, every recommendation is an epoch
                self.epoch_update()
                i += 1 

            except UserFullError as e:
                # As this is a loop for a fixed user, if the user is full no more recs can be done. Therefore, the loop ends
                e.error_msg()
                if enable_metrics:
                    outputDF = pd.DataFrame(resultsL, columns=['user_id', 'item_id', 'received_reward', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering']) 
                else:
                    outputDF = pd.DataFrame(resultsL, columns=['user_id', 'item_id', 'received_reward']) 
                return outputDF
 
        if enable_metrics:
            outputDF = pd.DataFrame(resultsL, columns=['user_id', 'item_id', 'received_reward', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering']) 
        else:
            outputDF = pd.DataFrame(resultsL, columns=['user_id', 'item_id', 'received_reward']) 

        return outputDF

