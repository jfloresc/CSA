import unittest
import numpy as np
import os
import tempfile
from spin import Spin, CSA
from randcsa import RandCSA

class TestRandCSA(unittest.TestCase):
    def setUp(self):
        self.rand_csa = RandCSA(randomseed=42)

    def test_initialization(self):
        self.assertIsInstance(self.rand_csa._random, np.random.Generator)

    def test_get_state(self):
        state = self.rand_csa.get_state
        self.assertIsInstance(state, dict)

    def test_set_state(self):
        initial_state = self.rand_csa.get_state
        # Generate a random number to change the state
        self.rand_csa.random.random()
        # Get the new state
        new_state = self.rand_csa.get_state
        # Ensure the state has actually changed
        self.assertNotEqual(initial_state, new_state)
        # Now set the state back to the initial state
        self.rand_csa.set_state = initial_state
        # Check that the state has been reset
        self.assertEqual(self.rand_csa.get_state, initial_state)

    def test_random_property(self):
        self.assertIsInstance(self.rand_csa.random, np.random.Generator)

    def test_save_load_state(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            initial_state = self.rand_csa.get_state
            self.rand_csa.save(tmp.name)
            self.rand_csa.random.random()  # Change the state
            self.rand_csa.load(tmp.name)
            loaded_state = self.rand_csa.get_state
        self.assertEqual(initial_state, loaded_state)
        os.unlink(tmp.name)

class TestSpin(unittest.TestCase):
    def setUp(self):
        self.n_size = 31
        self.serial_index = 0
        self.spin = Spin(self.n_size, self.serial_index, randomseed=42)
        self.spin.initialize()

    def test_initialization(self):
        self.assertEqual(self.spin._n_size, self.n_size)
        self.assertEqual(self.spin._id, self.serial_index)
        self.assertIsInstance(self.spin._prng, RandCSA)
        self.assertEqual(self.spin._j_coupling.shape, (self.n_size, self.n_size))

    def test_hamiltonian(self):
        spins = np.ones(self.n_size)
        energy = self.spin.hamiltonian(spins)
        self.assertIsInstance(energy, float)

    def test_minimize(self):
        spins = self.spin._prng.random.choice([-1, 1], size=self.n_size)
        minimized_spins, minimized_energy = self.spin.minimize(spins)
        self.assertEqual(len(minimized_spins), self.n_size)
        self.assertIsInstance(minimized_energy, float)

    def test_distance(self):
        obj1 = np.ones(self.n_size)
        obj2 = -np.ones(self.n_size)
        distance = self.spin.distance(obj1, obj2)
        self.assertEqual(distance, min(self.n_size, self.n_size - self.n_size))  # Should be 0

    def test_crossover(self):
        obj1 = np.ones(self.n_size)
        obj2 = -np.ones(self.n_size)
        
        for _ in range(10):  # Try multiple times due to potential randomness
            crossed = self.spin.crossover(np.copy(obj1), obj2)
            self.assertEqual(len(crossed), self.n_size)
            print(f"Test crossover original: {obj1}")
            print(f"Test crossover modified: {crossed}")
            
            if np.any(crossed != obj1):
                # Crossover changed something, test passes
                break
        else:
            # If we get here, crossover didn't change anything in 10 attempts
            print(f"Original obj1: {obj1}")
            print(f"Original obj2: {obj2}")
            print(f"Crossed result: {crossed}")
            self.fail("Crossover did not change the object after 10 attempts")


    def test_mutation(self):
        obj = np.ones(self.n_size)
        for _ in range(10):  # Try multiple times due to randomness
            mutated = self.spin.mutation(np.copy(obj))
            self.assertEqual(len(mutated), self.n_size)
            if np.any(mutated != obj):
                break
        else:
            self.fail("Mutation did not change the object after 10 attempts")

class TestCSA(unittest.TestCase):
    def setUp(self):
        self.n_init_bank = 15
        self.n_seed = 6
        self.csa = CSA(self.n_init_bank, self.n_seed)
        self.spin = Spin(10, 0, randomseed=42)
        self.spin.initialize()
        self.csa.set_gmin(self.spin)

    def test_initialization(self):
        self.assertEqual(self.csa._n_init_bank, self.n_init_bank)
        self.assertEqual(self.csa._n_seed, self.n_seed)

    def test_createbank(self):
        bank, ene_max, i_max = self.csa.createbank(self.n_init_bank)
        self.assertEqual(len(bank), self.n_init_bank)
        self.assertIsInstance(ene_max, float)
        self.assertIsInstance(i_max, int)

    def test_findminimum(self):
        bank, _, _ = self.csa.createbank(self.n_init_bank)
        ene_min, i_min = self.csa.findminimum(bank)
        self.assertIsInstance(ene_min, float)
        self.assertIsInstance(i_min, int)

    def test_average_dist(self):
        bank, _, _ = self.csa.createbank(self.n_init_bank)
        avg_dist = self.csa.average_dist(bank)
        self.assertIsInstance(avg_dist, float)

    def test_get_seeds(self):
        bank, _, _ = self.csa.createbank(self.n_init_bank)
        record = [False] * len(bank)
        seeds = self.csa.get_seeds(bank, record, False)
        self.assertEqual(len(seeds), self.n_seed)

if __name__ == '__main__':
    unittest.main()
