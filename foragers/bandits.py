import numpy as np

class BernoulliBandit:
    # accepts a list of K >= 2 floats , each lying in [0 ,1]
    def __init__(self, means):
        self.means = means
        self.mu_star = np.max(self.means)
        self._regret = 0

    # Function should return the number of arms
    def K(self):
        return len(self.means)

    # Accepts a parameter 0 <= a <= K -1 and returns the
    # realization of random variable X with P ( X = 1) being
    # the mean of the ( a +1) th arm .
    def pull(self, a):
        outcome = np.random.binomial(1, self.means[a])
        self._regret += self.mu_star - self.means[a]
        return outcome 

    # Returns the regret incurred so far
    def regret(self):
        return self._regret