import numpy as np


class ManualDensity:
    """
    Class for handling density functions

    """
    def __init__(self):
        """
        Constuctor

        """
        pass

    def random_density(self, points):
        """
        Random density function for testing purposes

        :param points: set of landmarks in :math:`\mathbb{R}^d`.
        :type points: np.array size of *n_landmarks x 3*

        """
        return np.array(np.random.rand(1, len(points))[0])

    # TODO
    # find out other useful density functions
    # implement them
