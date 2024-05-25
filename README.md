# README #

Python 3 scripts. Serial implementation of Conformational Space Annealing (CSA), a global optimization method.
This method consists of a distance dependent simulated annealing and an evolutionary algorithm (see references below).

This major rewrite allows for the implementation of new Hamiltonian and objective functions using
a Base class as template. Crossover and mutations are designed according to the problem to solve. The same for the distance function between solutions.

A random generator class was implemented to facilitate future parallel CSA.

The Spin class reproduces results from:
- Seung-Yeon Kim, Sung Jong Lee, and Jooyoung Lee, Ground-state energy and energy landscape of the Sherrington-Kirkpatrick spin glass, 
Phys.Rev.B, Vol. 76, 184412-1 - 184412-7 (2007).

This is my open source implementation of CSA as described by (and references within):

- J. Lee, H. A. Scheraga, and S. Rackovsky, J. Comput. Chem. Vol. 18,
1222, (1997)

- J. Insuk et al., Computer Physics Communications, Vol. 223, 28-33, (2018).

TODO:
Parallel implementation of CSA global optimization method.

Written by Jose Flores-Canales
