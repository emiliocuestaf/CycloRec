class Error(Exception):
    """Base class for exceptions in this module."""
    pass

# Recommendation round exceptions

class UserFullError(Error):
    """Exception raised when trying to recommendate to an user which has already been recommended all the items"""
    def __init__(self, user=None):
        self.user = user

    def error_msg(self):
        if self.user != None:
            print("\t\tThe user {} can't be recommended more items".format(self.user))
        else:
            print("\t\tThe user can't be recommended more items")
        return 

class SystemFullError(Error):
    """Exception raised when the system detects that all the users have been recommended all the items (Stop condition)"""
    def error_msg(self):
        print("\t\tAll the possible recommendations have been made!")
        return

# KNN Exceptions

class NoPositiveFeedbackError(Error):
    """Exception raised when trying an item based knn over an user that doesnt have positive ratings"""
    def error_msg(self):
        print("NoPositiveFeedbackERROR: The user doesnt have any positive feedback")
        return

class UnknownDistance(Error):
    """Exception raised when trying to calculate neighbors for an item that doesnt have valorations"""
    def error_msg(self):
        print("UnknownDistanceERROR: The element doesnt have any non 0 feedback")
        return

# MAB Errors

class UnavailableArms(Error):
    """ Exception raised when we want a MAB to choose an arm but all of them are blocked."""
    def error_msg(self):
        print("UnavailableArmsERROR: The MAB doesnt have any pullable arm")
        return