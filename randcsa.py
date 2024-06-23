###################################################################################################
#Copyright (c) 2022-2024 Jose Flores-Canales
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#associated documentation files (the "Software"), to deal in the Software without restriction,
#including without limitation the rights to use, copy, modify, merge, publish, distribute,
#sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or
#substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###################################################################################################

"""
Declaration of RandCSA class to handle random generators.
Using default bit generator PCG64.
Consider using bit generator PCG64DXSM for a parallel implementation.
SeedSequence is needed for multiple concurrent processes.
"""

import numpy as np
import pickle

class RandCSA(object):
    """Define RandCSA methods"""
    def __init__(self, randomseed=None):
        """Create random generator object"""
        self._rand_seed = randomseed
        self._random = np.random.default_rng(self._rand_seed)

    @property
    def get_state(self):
        """get state"""
        return self._random.bit_generator.state

    @property
    def set_state(self):
        """set state"""
        return self._random.bit_generator.state

    @set_state.setter
    def set_state(self, state):
        """set state"""
        self._random.bit_generator.state = state 

    @property
    def random(self):
        """get random state"""
        return self._random

    @random.setter
    def random(self, random):
        """get random state"""
        if isinstance(random, np.random.RandomState):
            self._random = random
        else:
            raise TypeError("Type must be a random.RandomState")

    def save(self, filename):
        """save random state"""
        state = self._random.bit_generator.state
        with open(filename, 'wb') as fh:
            pickle.dump(state, fh)

    def load(self, filename):
        """load random state"""
        with open(filename, 'rb') as fh:
            self._random.bit_generator.state = pickle.load(fh)
