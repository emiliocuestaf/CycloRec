 
class Evaluator():

    def __init__(self, dataLayer):
        """ Constructor of the class Evaluator

        Args:
            dataLayer (dataLayer): Instance of DataLayer, used to get the relevant and antirelevant counts and 
                                   to check if a passed ratings is relevant or antirelevant respectively.
        """
        super().__init__()

        self.dataLayer = dataLayer
        self.n_iters = 0

        self.cum_recall_value = 0
        self.recall_count = 0

        self.cum_fallout_value = 0
        self.fallout_count = 0
        
        self.precision_count = 0
        self.cum_precision_value = 0
        
        self.antiprecision_count = 0
        self.cum_antiprecision_value = 0
        
        self.cum_discovering = 0
        self.discovering_count = 0



    ###########################################################
    ###                                                     ###
    ###                   ALL METRICS                       ###
    ###         Calculates/updates 4 different metrics      ###
    ###             and the discovering rate                ###
    ###                                                     ###    
    ###########################################################


    def all_metrics(self, rating):
        """ Updates and stores the cumulative values for recall, precision, fallout and antiprecision and the discovering rate. 

        Args:
            rating (float): Rating of the last performed recommendation

        Returns:
            [float, float, float, float, float]: cumulative recall, cumulative precision, cumulative fallout, cumulative antiprecision, discovering rate
        """

        self.n_iters += 1
        toggle = 0
        # CLASSIC METRICS
        if self.dataLayer.isRelevant(rating):
            toggle = 1
            # RECALL
            self.recall_count += 1
            self.cum_recall_value = self.recall_count/self.dataLayer.get_relevant_count()
            # PRECISION
            self.precision_count += 1

        self.cum_precision_value = self.precision_count/self.n_iters

            
        # FALSE POSITIVE METRICS
        if self.dataLayer.isAntiRelevant(rating):
            toggle = 1
            # FALLOUT
            self.fallout_count += 1
            self.cum_fallout_value = self.fallout_count/self.dataLayer.get_antirelevant_count()
            # ANTIPRECISION
            self.antiprecision_count += 1
        
        self.cum_antiprecision_value = self.antiprecision_count/self.n_iters


        # RATIO OF DISCOVERING
        if  toggle:
            # This measure is useful to check if the algorithm is discovering any kind of important information
            # (it could be recommending items without any valorations in test all the time)
            self.discovering_count += 1
            self.cum_discovering = self.discovering_count/self.dataLayer.get_toDiscover_count()

        return self.cum_recall_value, self.cum_precision_value, self.cum_fallout_value, self.cum_antiprecision_value,  self.cum_discovering

    def get_metrics(self):
        return self.cum_recall_value, self.cum_precision_value, self.cum_fallout_value, self.cum_antiprecision_value,  self.cum_discovering


    ###########################################################
    ###                                                     ###
    ###              OTHER FUNCTIONALITIES                  ###
    ###                                                     ###
    ###########################################################


    def restart_metrics(self):
        """ Restarts the Evaluator to an initial state.  """
        self.n_iters = 0
        self.cum_recall_value = 0
        self.cum_fallout_value = 0
        self.precision_count = 0
        self.cum_precision_value = 0
        self.antiprecision_count = 0
        self.cum_antiprecision_value = 0
        self.cum_discovering = 0