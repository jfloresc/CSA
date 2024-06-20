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
Script that performs conformational space annealing.
it reproduces results from
Seung-Yeon Kim, Sung Jong Lee, and Jooyoung Lee , Ground-state energy and energy landscape of
the Sherrington-Kirkpatrick spin glass, Phys.Rev.B, Vol. 76, 184412-1 - 184412-7 (2007).
"""

import os
import copy
import json
import abc
import numpy as np
from randcsa import RandCSA


###################################################################################################
# Classes
###################################################################################################

class Base(metaclass=abc.ABCMeta):
    """Define abstract methods to be used by CSA objects and data types to optimize"""
    @abc.abstractmethod
    def createbank(self, size_bank):
        """Create bank method"""

    @abc.abstractmethod
    def minimize(self, obj):
        """Minimize Object"""

    @abc.abstractmethod
    def distance(self, obj_1, obj_2):
        """Pairwise distance between two objects"""

    @abc.abstractmethod
    def crossover(self, obj_1, obj_2):
        """Merge operation between two objects"""

    @abc.abstractmethod
    def crossover1(self, obj_1, obj_2):
        """Merge operation between two objects"""

    @abc.abstractmethod
    def mutation(self, obj_1):
        """Mutates in object"""


class Spin(Base):
    """Define data type spin
    Template for new data types
    """

    def __init__(self, n_size, serial_index, randomseed=None):
        """Create new spin object
        """
        self._n_size = n_size
        self._id = serial_index
        self._rand_seed = randomseed
        self._j_coupling = np.array([])
        self._name = None 
        self._prng = None

    def initialize(self):
        """Initialize object"""
        self.start_rand_state()
        self.initiate_j_coupling()

    def copy(self):
        """Return a copy of the data type object"""
        out = Spin(self._n_size, self._id, self._rand_seed)
        out.set_j_coupling(self._j_coupling)
        out.name(self._name)
        return out

    def __copy__(self, *args):
        """Magic method implementation"""
        return self.copy()

    def __deepcopy__(self, *args):
        """Magic method implementation"""
        return self.copy()

    def __len__(self):
        """Magic method implemntation"""
        return self._j_coupling.shape[0]

    def set_j_coupling(self, j_coupling):
        """Set J coupling"""
        self._j_coupling = j_coupling

    @property
    def name(self):
        """get name of J coupling file"""
        return self._name

    @name.setter
    def name(self, name):
        """Set name of J coupling file"""
        self._name = name

    @property
    def id(self):
        """Get serial index of spin object"""
        return self._id

    def start_rand_state(self):
        """Create a randomstate instance"""
        self._prng = RandCSA(self._rand_seed)

    def initiate_j_coupling(self):
        """Create a new spin state"""
        self._j_coupling = self._prng.random.choice([-1,1], size=(self._n_size, self._n_size))
        self._name = f'J_{self._id}'

    def write_j_coupling(self):
        """save state in file"""
        np.save(self._name, self._j_coupling)

    def read_j_coupling(self, filename):
        """read state from file"""
        self._j_coupling = np.load(filename)

    def createbank(self, size_bank):
        """Create a bank
        Parameters
        ----------
        size_bank : number of spins vectors

        Returns
        -------
        bank : a list containing spin lists and their respective locally minimized energies
        ene_max : maximum energy in the bank
        imax : list index of spin list with the maximum energy
        """
        bank = []
        ene_max = -1000000*self._n_size
        # creating a list of size_bank random spin vectors of size n_size
        temp = self._prng.random.choice([-1,1], size=(size_bank, self._n_size))
        temp = [list(x) for x in temp]
        for i, random_spins in enumerate(temp):
            #random_spins = self.randomize()
            spins_min, ene_min = self.minimize(random_spins)
            # adds tuple of minimized list of spins and its respective energy
            bank.append([spins_min, ene_min])
            if ene_min > ene_max:
                ene_max = ene_min
                imax = i
        return bank, ene_max, imax

    def minimize(self, obj):
        """Local energy minimization function"""
        #init = hamiltonian(obj, j_coupling, n_size)
        temp = copy.deepcopy(obj)
        count = 0
        while count < self._n_size:
            for i in range(self._n_size):
                dene = self.dhamiltonian(temp, i)
                if dene < 0:
                    temp[i] = -temp[i]
                    count = 0
                else:
                    count += 1
                if count == self._n_size:
                    break
        ene = self.hamiltonian(temp)
        return temp, ene

    def randomize(self):
        """Get a randomized vector of spins: -1 or +1"""
        random_spins = []
        s_value = lambda p: -1 if p == 0 else 1
        for _ in range(self._n_size):
            p_int = self._prng.random.integers(2)
            random_spins.append(s_value(p_int))
        return random_spins

    def dhamiltonian(self, spins, idx):
        """Calculate change of hamiltonian"""
        energy_old = 0
        energy_new = 0

        # same hamiltonian, i,j != idx
        #  for i in range(n_size):
        #    if i == idx: continue
        #    for j in range(i+1,n_size):
        #      if j == idx: continue
        #      E += spins[i]*spins[j]*j_coupling[i,j]

        # changes for i = idx
        s_idx = spins[idx]
        for j in range(idx+1, self._n_size):
            energy_old += s_idx*spins[j] * self._j_coupling[idx,j]
            energy_new += -s_idx*spins[j] * self._j_coupling[idx,j]

        # changes for j = idx
        for i in range(idx):
            energy_old += spins[i] * s_idx * self._j_coupling[i,idx]
            energy_new += -spins[i] * s_idx * self._j_coupling[i,idx]
        return  float(-energy_new+energy_old)/(np.sqrt(self._n_size)*self._n_size)

    def hamiltonian(self, spins):
        """Calculate of hamiltonian function given parameters j_coupling
        and a configuration of spins"""
        energy = 0
        for i in range(self._n_size):
            for j in range(i+1, self._n_size):
                energy += spins[i]*spins[j]*self._j_coupling[i,j]
        return  -float(energy)/(np.sqrt(self._n_size)*self._n_size)

    def projection(self, spins_a, spins_b) :
        """Calculate projection"""
        q_value = 0
        for i,j in zip(spins_a, spins_b):
            q_value += i*j
        return abs(q_value)/float(self._n_size)

    def distance(self, obj_1, obj_2):
        """Distance function"""
        dist = 0
        for i, j in zip(obj_1, obj_2):
            if i != j:
                dist += 1
        return min(dist, self._n_size-dist)

    def crossover(self, obj_1, obj_2):
        """Merge spins from gene2 to gene1"""
        # maximum number of spins to be replaced at a time
        # up to 50 % of the total number of variables
        max_vars = int(self._n_size/2.)
        # number of variables to replace
        k = self._prng.random.integers(1, max_vars + 1)
        # indexes of the variables to alter
        #rindx = random.sample(range(self._n_size), k)
        rindx = self._prng.random.choice(range(self._n_size), k)
        # temporary solution when the base dataset is numpy array
        if type(obj_2) == np.ndarray:
            obj_1[rindx] = obj_2[rindx]
        elif type(obj_2) == list:
            for idx in rindx:
                obj_1[idx] = obj_2[idx]
        return obj_1

    def crossover1(self, obj_1, obj_2):
        """Merge spins from gene2 to gene1"""
        # maximum number of spins to be replaced at a time
        # up to 20 % of the total number of variables
        max_vars = int(self._n_size/5.)
        # number of variables to replace
        k = self._prng.random.integers(1, max_vars + 1)
        # indexes of the variables to alter
        #rindx = random.sample(range(self._n_size), k)
        rindx = self._prng.random.choice(range(self._n_size), k)
        # temporary solution when the base dataset is numpy array
        if type(obj_2) == np.ndarray:
            obj_1[rindx] = obj_2[rindx]
        elif type(obj_2) == list:
            for idx in rindx:
                obj_1[idx] = obj_2[idx]
        return obj_1

        '''
    def mutation(self, obj_1):
        """Mutate gene"""
        # number of variables to mutate
        # up to 3 variables are randomized
        k = np.random.randint(1,4)
        # indexes of the variables to alter
        #rindx = random.sample(range(self._n_size), k)
        rindx = self._prng.choice(range(self._n_size), k)
        # temporary solution when the base dataset is numpy array
        if type(obj_1) == np.ndarray:
            # values to randomize -1 or +1
            obj_1[rindx] = self._prng.choice([-1,1], size=k) 
            # negating
            #obj_1[rindx] = -obj_1[rindx] 
        elif type(obj_1) == list:
            for idx in rindx:
                obj_1[idx] = self._prng.choice([-1, 1]) 
        return obj_1
        '''

    def mutation(self, obj_1):
        """Mutate gene"""
        # number of variables to mutate
        # up to 3 variables are randomized
        # percentage 
        m = int(self._n_size * 0.05)
        # m = 3
        k = self._prng.random.integers(1, m + 1)
        # indexes of the variables to alter
        idx = self._prng.random.choice(range(self._n_size - k - 1))
        # temporary solution when the base dataset is numpy array
        if type(obj_1) == np.ndarray:
            # values to randomize -1 or +1
            obj_1[idx:idx+k] = self._prng.random.choice([-1,1], size=k) 
            # negating
            #obj_1[rindx] = -obj_1[rindx] 
        elif type(obj_1) == list:
            obj_1[idx:idx+k] = [self._prng.random.choice([-1, 1]) for x in range(k)]
        return obj_1

class CSA(Base):
    """CSA object defines methods and attributes of a CSA object.
    CSA class is designed to be associated with any type of data type also derevied
    from the Base clase.
    Attributes
    ----------
    nbank1
    bank
    """
    # pylint: disable=too-many-instance-attributes
    #CALLS, 15, 6, N_ROUNDS, 10000
    def __init__(self,\
                n_init_bank,\
                n_seed,\
                n_daughters=30,\
                n_rounds=27,\
                n_max_trials=10000):
        """Create new CSA object"""
        self._n_init_bank = n_init_bank
        self._n_new_config = n_init_bank
        self._n_seed = n_seed
        self._n_daughters = n_daughters
        self._n_rounds = n_rounds
        self._n_max_trials = n_max_trials
        self._trials = 0
        self._k = 0.0
        #self._instance_counter = 0
        self._dist_traj_file = ''
        self._global_ene_file = ''
        self._gmin = None
        self._dcut = 0
        self._dave = 0

    @property
    def id(self):
        """Get serial index of CSA object"""
        return self._id

    @id.setter
    def id(self, serial_index):
        """Set serial index of CSA object"""
        self._id = serial_index

    def set_gmin(self, myobject):
        """Add object to optimize globably"""
        self._gmin = myobject
        # takes id serial number from data object to minimize
        self.id = myobject.id

    def initialize(self):
        """Initiates parameters and files"""
        # calculates the ratio parameter 
        #self._k = np.power(0.4, 1./self._n_max_trials)
        # defined in the paper I. Lee, Comp. P. C. 2017
        self._k = 0.983912

    def create_log_files(self, istep):
        """Open distance trajectory and global minimum energy files"""
        self._dist_traj_file = f'output_{self._id}_{istep}.dat'
        self._global_ene_file = f'minimum_{self._id}_{istep}.dat'
        print(f"Opening files: {self._dist_traj_file}, {self._global_ene_file}")
        if os.path.exists(self._dist_traj_file) or os.path.exists(self._global_ene_file):
            print(f"Files {self._dist_traj_file} {self._global_ene_file} were overwritten")
            with open(self._dist_traj_file, 'w', ) as _:
                pass
            with open(self._global_ene_file, 'w', ) as _:
                pass

    def log_traj_file(self):
        """Log distance trajectory"""
        with open(self._dist_traj_file, 'a', ) as stream:
            stream.write('%6d %5.3f\n' % (self._trials, self._dcut))

    def log_gmin_file(self, istep, bank_ene_min):
        """Log global minimum energy"""
        with open(self._global_ene_file, 'a', ) as estream:
            estream.write('%6d %5.6f\n' % (istep, bank_ene_min))

    def createbank(self, size_bank):
        """Execute the createbank of the object to optimize
        Parameters
        ----------
        nbank : number of spins vectors
        """
        return self._gmin.createbank(size_bank)

    def encoder_np(self, obj):
        """temporary solution before converting everything to numpy"""
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.item()

    def print_bank(self, name, bank):
        """Print bank in json format"""
        with open(name, 'w', ) as stream:
            json.dump(bank, stream, default=self.encoder_np)
            stream.flush()

    def findminimum(self, bank):
        """Find object with the minimum energy in the bank
        bank is a list of tuples
        each tuple contains an object and its associated energy value"""
        ene_min = bank[0][1]
        i_min = 0
        for i, obj in enumerate(bank):
            if obj[1] < ene_min:
                ene_min = obj[1]
                i_min = i

        return ene_min, i_min

    def findmaximum(self, bank):
        """Find object with the maximum energy in the bank
        bank is a list of tuples
        each tuple contains an object and its associated energy value"""
        ene_max = bank[0][1]
        i_max = 0
        for i, obj in enumerate(bank):
            if obj[1] > ene_max:
                ene_max = obj[1]
                i_max = i

        return ene_max, i_max

    def average_dist(self, bank):
        """Get average distance
        bank is a list of tuples (object, object_energy)"""
        nbank = len(bank)
        count, sum_dist = 0, 0
        for i in range(nbank):
            for j in range(i+1, nbank):
                sum_dist += self.distance(bank[i][0], bank[j][0])
                count += 1
        if count == 0:
            raise Exception("count variable is zero, check the size of bank")

        return float(sum_dist)/count

    def distance(self, obj_1, obj_2):
        """Get distance between two objects"""
        return self._gmin.distance(obj_1, obj_2)

    def minimize(self, obj):
        """Local energy minimization function"""
        # update the number of trials for every minimization carried out
        self._trials += 1
        return self._gmin.minimize(obj)

    def crossover(self, obj_1, obj_2):
        """Crossover with bank"""
        return self._gmin.crossover(obj_1, obj_2)

    def crossover1(self, obj_1, obj_2):
        """Crossover with bank1"""
        return self._gmin.crossover1(obj_1, obj_2)

    def mutation(self, obj_1):
        """Mutate"""
        return self._gmin.mutation(obj_1)

    def find_next_seed(self, first_seed, bank, record_index, n_missing_seeds, seeds_tuples, selected):
        """Auxiliary function of get_seeds()"""
        for i_seed in range(n_missing_seeds):
            index_to_d_energy = {}
            for i in record_index:
                # different from the selected seed
                if i in selected:
                    continue
                dij = self.distance(first_seed, bank[i][0])
                # dictionary with bank conformation index as key to value of a
                # tuple of distance to the conformation and conformation's energy
                index_to_d_energy[i] = (dij, bank[i][1])
            # filter bank conformations with distance from current seed > dave
            temp = {k: v for k, v in index_to_d_energy.items() if v[0]  > self._dave}
            if temp:
                # find index of conformation with minimum energy after the filter
                sel_idx = min(temp.items(), key=lambda k:k[1][1])[0]
            # if filter result is zero, select seed with largest distance
            else:
                sel_idx = max(index_to_d_energy.items(), key=lambda k:k[1][0])[0]
            selected.append(sel_idx)
            seeds_tuples.append((sel_idx, copy.deepcopy(bank[sel_idx][0])))
        return seeds_tuples

    def get_seeds(self, bank, record, is_new_stage):
        """selects seeds and returns a list of tuples: (index, seed)"""
        r_bank = range(len(record))
        seeds_tuples = []
        # get indexes of unused spin vectors
        if is_new_stage:
            # get indexes of only the newly added configurations to the bank in the new stage
            #record_index = [i for i, x in zip(r_bank[-self._n_new_config:], record[-self._n_new_config:]) if x == False]
            record_index = [i for i in r_bank[-self._n_new_config:] if not record[i]]
        else:
            #record_index = [i for i, x in enumerate(record) if x == False]
            record_index = [i for i, x in enumerate(record) if not x]
        n_unused = len(record_index)
        if n_unused >= self._n_seed:
            print(f"First condition applies, #n_unused {n_unused}, #n_seed {self._n_seed}")
            # get the index of an unused spin vector from the bank
            idx = self._gmin._prng.random.choice(record_index)
            first_seed = copy.deepcopy(bank[idx][0])
            seeds_tuples.append((idx, first_seed))
            selected = [idx]
            # search for the ( n_seed - first_seed ) unused seeds
            n_missing_seeds = self._n_seed - 1
            seeds_tuples = self.find_next_seed(first_seed, bank, record_index, n_missing_seeds, seeds_tuples, selected)
        else:
            print(f"selecting all the unused, #n_unused {n_unused}, #n_seed {self._n_seed}")
            # select all unused
            seeds_tuples = [(idx, bank[idx][0]) for idx in record_index]
            selected = record_index 
            # repeat the same as the above condition but on the used spin vectors
            # get indexes of used spin vectors
            record_index_used = [i for i, x in enumerate(record) if x == True]
            # get the index of an unused spin vector from the bank
            idx = self._gmin._prng.random.choice(record_index_used)
            first_seed = copy.deepcopy(bank[idx][0])
            seeds_tuples.append((idx, first_seed))
            # search for the ( n_seed - selected used seeds - first_seed )
            n_missing_seeds = self._n_seed - len(selected) -1 
            seeds_tuples = self.find_next_seed(first_seed, bank, record_index_used, n_missing_seeds, seeds_tuples, selected)
        return seeds_tuples

    def generateconfig(self, bank, bank1, record, is_new_stage):
        """Generate daughter configurations"""
        r_bank1 = range(len(record))
        # list of tuples containing minimized daughters 
        genes_ene = []
        # select n_seed random configrations from the bank
        # returns a list of tuples: index, seed 
        #seeds_tuples = random.sample(list(enumerate(bank)), self._n_seed)
        # select n_seed configurations from the bank using a defined strategy
        seeds_tuples = self.get_seeds(bank, record, is_new_stage)

        seed_counter = 0
        seedorder = list(range(self._n_seed))
        self._gmin._prng.random.shuffle(seedorder)
        #for index, seed in seeds_tuples:
        for rseed in seedorder:
            index, seed = seeds_tuples[rseed]
            seed_counter += 1
            print(f"#seed {seed_counter} Seed index: {index}")
            if is_new_stage:
                # choose from the newly added configurations
                #record_index = [i for i, x in enumerate(record[self._n_new_config:]) if x == False and i != index]
                #record_index = [i for i, x in zip(r_bank1[-self._n_new_config:], record[-self._n_new_config:])\
                #                if x == False and i != index]
                record_index = [i for i, x in zip(r_bank1[-self._n_new_config:], record[-self._n_new_config:]) if i != index]
                #print('First round in stage', record_index)
            else:
                #record_index = [i for i, x in enumerate(record) if x == False and i != index]
                record_index = [i for i, x in enumerate(record) if i != index]
                #print('next rounds in a stage', record_index)

            # randomize the operations
            # this can be parallelized
            randorder = list(range(self._n_daughters))
            self._gmin._prng.random.shuffle(randorder)
            for count in randorder:
                # each method is performed 10 times for n_daughters = 30
                # 10 is hardwired better change to constant variable
                mod = count // 10
                # select from bank1
                if mod == 0:
                    if is_new_stage:
                        # choose from the newly added configurations
                        idx = self._gmin._prng.random.choice(r_bank1[-self._n_new_config:])
                    else:
                        idx = self._gmin._prng.random.choice(r_bank1)
                    gen = self.crossover1(seed, bank1[idx][0])
                    gen_min, ene_min = self.minimize(gen)
                    genes_ene.append([gen, ene_min])
                # select from the bank
                elif mod == 1:
                    # get indexes of unused spin vectors and different from the current seed
                    # get the index of an unused spin vector from the bank
                    idx = self._gmin._prng.random.choice(record_index)
                    gen = self.crossover(seed, bank[idx][0])
                    gen_min, ene_min = self.minimize(gen)
                    genes_ene.append([gen, ene_min])
                    # mark spin vector as taken
                    #record[idx] = True
                # mutate
                else:
                    gen = self.mutation(seed)
                    gen_min, ene_min = self.minimize(gen)
                    genes_ene.append([gen, ene_min])
            # mark spin vector as taken
            record[index] = True
                
        return genes_ene, record

    def updatebank(self, genes_ene, bank):
        """Function to update bank"""
        nbank = len(bank)
        ene_max, i_max = self.findmaximum(bank)
        if genes_ene[1] >= ene_max:
            return bank
        # rolling out first index
        dij_min = self.distance(genes_ene[0], bank[0][0])
        j_min = 0
        for j in range(1, nbank):
            # calculate distance between daugther and each bank configuration
            dij = self.distance(genes_ene[0], bank[j][0])
            # update minimum distance
            if dij < dij_min:
                dij_min = dij
                j_min = j

        # check the paper about the rules here
        # bug, originally was dij_min < dcut
        # insert the new global minimum
        if dij_min <= self._dcut and genes_ene[1] < bank[j_min][1]:
            #bank[j_min][0] = copy.deepcopy(genes_ene[0])
            #bank[j_min][1] = copy.deepcopy(genes_ene[1])
            bank[j_min] = genes_ene
        # update the local minimum with the highest energy energy
        elif dij_min > self._dcut and genes_ene[1] < ene_max:
            #print('update ene_max', ene_max, 'new_ene:', genes_ene[1])
            #bank[i_max][0] = copy.deepcopy(genes_ene[0])
            #bank[i_max][1] = copy.deepcopy(genes_ene[1])
            bank[i_max] = genes_ene

        return bank

    def calc(self, istep):
        """Calculate global minimum"""

        # create or rewrite log files
        self.create_log_files(istep)

        # create local variable
        nbank = self._n_init_bank
        # create first bank: bank1
        # get seed with the maximum energy and corresponding index location
        bank1, ene_max1, i_max = self.createbank(nbank)
        # saves the first bank in a file
        name = f'bank1_{self._id}_{istep}.dat'
        self.print_bank(name, bank1)

        # create copy of the first bank: bank
        bank = copy.deepcopy(bank1)
        ene_max, bank_max = self.findmaximum(bank)

        # calculate initial average distance (dav) and dcut from the first bank1
        self._dave = self.average_dist(bank1)
        self._dcut = self._dave / 2.0
        print(f'nbank: {nbank}\nnseed: {self._n_seed}\ndcut: {self._dcut}\nratio:'
            f'{self._k}')

        # get minimum energy object configuration for each bank and their index location
        bank1_ene_min, bank1_i_min = self.findminimum(bank1)
        bank_ene_min, bank_i_min = self.findminimum(bank)

        print(f'CSA starting conditions')
        print(f'i_bank_max: {bank_max} Max. Ene_max: {ene_max} Min. Ene_min: {bank_ene_min}'\
              f' i_bank_min: {bank_i_min} nbank: {nbank} i_bank1_max: {i_max} Ene_max1: '\
              f'{ene_max1} Ene_min1: {bank1_ene_min} i_bank1_min: {bank1_i_min}')

        # iterate over the total number of rounds, 3 x number of stages
        iround = 0 
        print(f'Initiation loop with {self._n_rounds}')
        while iround < self._n_rounds:
            # every three rounds expands both bank1 and nbank by n_new_config
            is_new_stage = True if iround % 3 == 0 and iround != 0 else False

            # changing stage
            # stage = stage + 1
            # expand both the first bank and the bank
            # update dave and dcut
            # update local variables
            if is_new_stage:
                print(f'Iround {iround}')
                trial_bank, ene_max_t, i_max_t = self.createbank(self._n_new_config)
                # debugging 
                #print('after adding config. to bank, ene_max', ene_max,'ene_max_new', ene_max_t)
                print(f"For the new bank addition: Ene_max {ene_max_t}, at index {i_max_t}")
                trial_ene_min, trial_i_min = self.findminimum(trial_bank)
                print(f"For the new bank addition: Ene_min {trial_ene_min}, at index {trial_i_min}")

                # increase bank1 and bank
                # extending lists, for np.arrays use concatenate in axis=0
                # bank1 = bank1 + copy.deepcopy(trial_bank)
                bank1.extend(trial_bank)
                #bank = copy.deepcopy(bank) + copy.deepcopy(trial_bank)
                bank.extend(trial_bank)
                # both banks have changed their lenght by n_new_config
                nbank += self._n_new_config
                # adapt dave since bank1 has changed when a new stage begins
                #self._dave = self.average_dist(bank1)
                # adapt dcut to a half of dave
                self._dcut = self._dave / 2.0
                # for the first iteration of the stage only use the newly
                # generate configurations
                n_unused = self._n_new_config

                # locate minimum energy objects and their index for each bank
                bank_ene_min, bank_i_min = self.findminimum(bank)
                bank1_ene_min, bank1_i_min = self.findminimum(bank1)
                # locate maximum energy objects and their indeces for each bank
                ene_max, bank_max = self.findmaximum(bank)
                ene_max1, i_max = self.findmaximum(bank1)
                #print(f'CSA round/iteration: {iround}')
                print(f'i_bank_max: {bank_max} Max. Ene_max: {ene_max} Min. Ene_min:'\
                    f' {bank_ene_min} i_bank_min: {bank_i_min} nbank: {nbank}'\
                    f' i_bank1_max: {i_max} Ene_max1:'\
                    f' {ene_max1} Ene_min1: {bank1_ene_min} i_bank1_min: {bank1_i_min}')
            else:
                n_unused = len(bank)

            record = [False for i in range(len(bank))]
            # this condition is hard-wired, better define a constant variable
            n_unused_t = n_unused
            while n_unused_t > 10:
                # generates all daughters configurations, n_seed * 30
                print(f'Unused configurations {n_unused_t}')
                genes_ene, record = self.generateconfig(bank, bank1, record, is_new_stage)
                # count the number of unused bank configurations
                if is_new_stage:
                    n_unused_t = n_unused - sum(record[-self._n_new_config:])
                else:
                    n_unused_t = n_unused - sum(record)
                print(f'Unused configurations after generating configurations {n_unused_t}')
                # iterates over tuples
                for gen_i in genes_ene:
                    # update the bank
                    bank = self.updatebank(gen_i, bank)
            # end of while loop
            # one round of CSA of iteration was completed
            # log dcut
            self.log_traj_file()
            # first approach using the number of minimizations
            #if self._trials < self._n_max_trials:
            # second approach, see I. Joung, 2017. For now hard-wired. 
            if iround < 25:
                self._dcut = self._dcut*self._k
            else:
                self._dcut = self._dave / 5.0
            #print('after update, ene_max', ene_max)
            #print('trials, dcut: ',trials, dcut)

            # wrapping up round by printing curent status
            # locate minimum energy objects and their index for each bank
            bank_ene_min, bank_i_min = self.findminimum(bank)
            bank1_ene_min, bank1_i_min = self.findminimum(bank1)
            # locate maximum energy objects and their indeces for each bank
            ene_max, bank_max = self.findmaximum(bank)
            ene_max1, i_max = self.findmaximum(bank1)
            print(f'CSA round/iteration: {iround}')
            print(f'i_bank_max: {bank_max} Max. Ene_max: {ene_max} Min. Ene_min:'\
                f' {bank_ene_min} i_bank_min: {bank_i_min} nbank: {nbank}'\
                f' i_bank1_max: {i_max} Ene_max1:'\
                f' {ene_max1} Ene_min1: {bank1_ene_min} i_bank1_min: {bank1_i_min}')
            iround += 1
        print(f'Total number of minimizations: {self._trials}')
        gmin_ene, i_gmin = self.findminimum(bank)
        return (gmin_ene, i_gmin, bank)


    def run(self, istep):
        """Run CSA
            input:
                instances: number of cycles
                icall: number of repeats
        """

        gmin_ene, bank_i_gmin, gbank = self.calc(istep)
        print(f'Min from bank {gmin_ene}')
        self.log_gmin_file(istep, gmin_ene)
        name = f'bank_gmin_{self._id}_{istep}.dat'
        self.print_bank(name, gbank)



#def test_hamiltonian_minimization(j_coupling, n_size):
#    '''function to test hamiltonian function and minimization'''
#    np.random.seed(100)
#    spins = initialize(n_size)
#    print('Energy', hamiltonian(spins, j_coupling, n_size))
#    _, ene_min = minimize(spins, j_coupling, n_size)
#    print('Min. Energy', ene_min)

if  __name__ == '__main__':

    # create spin object to minimize
    # number of spins
    N_SPINS = 31 

    # total number of rounds
    N_ROUNDS = 27 # 3 rounds * 9 stages
    #n_init_bank, n_seed, n_daughters=30, n_rounds=27, n_max_trials=10000


    for serial_id in range(1):
        # create object to globally optimize
        spin = Spin(N_SPINS, serial_id, randomseed=2)
        spin.initialize()
        print(f"Seed used {spin._prng.get_state}")
        spin._prng.save("state.pkl")
        # read J_0.dat
        #spin.read_j_coupling()
        spin.write_j_coupling()

        my_csa = CSA(15, 6)#, N_ROUNDS, 10000)
        # initialize parameters and open files or rewrite them
        my_csa.initialize()
        # set initial state of the object to minimize
        my_csa.set_gmin(spin)
        # run CSA
        my_csa.run(serial_id)
