import numpy as np


class ManualDensity:

    def foo(self, points):
        return np.array(np.random.rand(1, len(points))[0])
    
    #TODO
    # find out other useful density functions
    # implement them