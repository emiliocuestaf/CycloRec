## CycloREC
CycloREC is a Python3 library designed to simulate long-term cyclic recommendation processes from explicit datasets and collect metrics. 

There are some great and complete recommendation libraries in Python, such as [Surprise!](http://surpriselib.com/), but most of them are oriented to predict the numerical ratings a user might give to a concrete item. CycloREC does not care about numerical values, we only care if a recommendation has been accurate (positive feedback, from now on relevant), disgusting (negative feedback, from now on antiRelevant) or if the user has not reacted towards it (which we will call irrelevant). 

We assume we got a collection **U** of users, a collection **I**  of items, and a set **R** of ratings composed by <u, i, r_ui> triplets, where u belongs to **U**, i to **I** and r_ui is the rating u gave to i. Then, CycloREC is able to recommend different items to a concrete user once and again, to recommend one item to each of the users in **U** (full recommendation round) or to repeat this process _N_ times or until no more recommendations can be performed. Each user's feedback to each of the recommendations is dynamically obtained (from an **R** test subset) just after the recommendation is shown to her, just as if a real user reacted to the item.  

For every simulation, it is possible to collect some default metrics (recall, fallout, precision, antiprecision) or just getting the recommendation results. The presence of false-positive metrics (fallout, antiprecision), makes CycloREC specially suitable to measure if our recommenders need from many "bad recommendations" to achieve good results. 

CycloREC supports both 'log' and 'matrix' formats for the collected ratings. 
It is also possible to add new recommenders easily. 

## Motivation

This project came up as a solution for my Computer Science degree's final project at Universidad Autónoma de Madrid. Although I could have used other libraries I decided to start implementing things by myself to get a better idea of the RS world and this is the result. 

## Code Example

![Usage Example](/img/usage_example.png)

## Main functions

These functions define the main behaviour of CycleREC. They are methods from the BaseRecommender class, from which all the recommenders must inherit. Therefore, every recommender's performance in an interactive cycle is easy to measure. 

#### Training

* BaseRecommender.train()

        ABSTRACT Method to implement the default training behaviour. It should only be executed once at the beginning of the recommendation task.
        

#### Rounds with feedback improvement and metrics

* BaseRecommender.recommendation_round(enable_repetitions=False, enable_metrics=False)

            Makes a single recommendation for each target user and evaluates it in the same step. 
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
                pandas.DataFrame -- a Pandas DataFrame containing the recommendations. Columns = ['user_id', 'item_id']
                float -- (Only if enable_metrics == True) cumulative recall metric of the whole round based on the training set
                float -- (Only if enable_metrics == True) mean precision metric of the whole round based on the training set
                float -- (Only if enable_metrics == True) cumulative fallout metric of the whole round based on the training set
                float -- (Only if enable_metrics == True) mean antiprecision metric of the whole round based on the training set
                float -- (Only if enable_metrics == True) cumulative discovering of the whole round based on the training set

* BaseRecommender.recommendation_round_loop(maxiter, enable_repetitions=False, enable_metrics=False, verbose=False)
        
        Simulates and evaluates, one by one, N rounds of recommendations. The recommender can learn from each recommendation due to the
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
            [0] pd.DataFrame with the columns=['t', 'user_id', 'item_id']
            [1] (if enable_metrics == True) pd.DataFrame with the columns=['t', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering']


#### Rounds without feedback improvement and metrics

* BaseRecommender.straight_recommendation_round(enable_repetitions=False)
 
        Recommends a single item to each of the users that can still be recommended. 
            Keeps the training set updated to avoid repeating recommendations but IT DOES NOT UPDATE THE RECOMMENDATION MODEL.
            Therefore, this function is suitable to get a list of recommended items but it won't improve the recommender's performance. 
        
        Keyword Arguments:
            enable_repetitions {bool} -- If True items that have been recommended before can be recommended again to the same user. 
                                         Therefore, if True every user is guaranteed to have exactly one recommendation.
                                         If False, items can't be recommended twice to the same user. (default: {False})
        Raises:
            SystemFullError: Launched when there are no more users to be recommended
        
        Returns:
            pd.DataFrame -- a Pandas DataFrame containing the computed recommendations. Columns = ['user_id', 'item_id']
            
* BaseRecommender.straight_recommendation_round_loop(maxiter, enable_repetitions=False, verbose=False)

        Computes N (maxiter) rounds of recommendations, one after the other. The recommender is not improving at each epoch
        as it uses the straigh_recommendation_round() function.
        
        Arguments:
            maxiter {int} -- Number of rounds to simulate. If all the users are full before reaching the maxiter round, the loop stops.
        
        Keyword Arguments:
            enable_repetitions {bool} -- True: Same item can be recommended twice or more times 
                                         False: Each item can only be recommended once to each user (default: {False})
            verbose {bool} -- If True: Terminal will show messages when the whole recommendation process is 20%, 40%, 60%, 80% and 100% fulfilled.
                              If False: Anything will be shown (default: {False})

        
        Returns: pd.DataFrame with the columns=['t', 'user_id', 'item_id']

#### Fixed user behavior

* BaseRecommender.fixed_user_recommendation_loop(user_id, maxiter, enable_repetitions=False, verbose=False)

        Recommends and evaluates, one by one, N recommendations to the specified user.
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

        
        Returns: pd.DataFrame with the columns=['user_id', 'item_id', 'recall', 'precision', 'fallout', 'antiprecision', 'discovering']


* BaseRecommender.fixed_user_straight_recommendation_loop(user_id, maxiter, enable_repetitions=False, verbose=False)
            
        Recommends, one by one, N recommendations to the specified user.
            If
                1) N is bigger to the number of the items the user has NOT been recommended yet (R)
                AND
                2) enable_repetitions == True, 
            then the loop will stop at R iterations.
            It does not learn in the process.
            
        
        Arguments:
            user_id {[type]} -- User to be recommended
            maxiter {[type]} -- Number of maximum recommendations
        
        Keyword Arguments:
            enable_repetitions {bool} -- True: Same item can be recommended twice or more times 
                                         False: Each item can only be recommended once to each user (default: {False})
            verbose {bool} -- If True: Terminal will show messages when the whole recommendation process is 20%, 40%, 60%, 80% and 100% fulfilled.
                              If False: Anything will be shown (default: {False})

        
        Returns: pd.DataFrame with the columns=['user_id', 'item_id']


## Installation

A .yml file is provided to install the conda environment with every needed package.

## Tests

(pending)

## How to use?

(pending)

## Contribute

Please issue every problem you notice. I know some of the default recommenders scale up in a horrible way. Of course, I'll be pleased to add any performance improvement!

## Credits

[1] Pablo Castells and Javier Sanz-Cruzado, my project supervisors.

[2] [Nicolas Hug, Surprise! creator.](http://nicolas-hug.com/)
