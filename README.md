# packing-problem
Final project for Optimization Models Exam, lectured by prof. Castelli provided by Units.

Reproduction of the results of 

Allyson Silva, Leandro C. Coelho, Maryam Darvish, Jacques Renaud,

A cutting plane method and a parallel algorithm for packing rectangles in a circular container,

European Journal of Operational Research,

Volume 303, Issue 1,

2022,

Pages 114-128,

ISSN 0377-2217,

https://doi.org/10.1016/j.ejor.2022.02.023.

(https://www.sciencedirect.com/science/article/pii/S037722172200128X)

Abstract: We study a two-dimensional packing problem where rectangular items are placed into a circular container to maximize either the number or the total area of items packed. We adapt a mixed-integer linear programming model from the case with a rectangular container and design a cutting plane method to solve this problem by adding linear cuts to forbid items from being placed outside the circle. We show that this linear model allows us to prove optimality for instances larger than those solved using the state-of-the-art non-linear model for the same problem. We also propose a simple parallel algorithm that efficiently enumerates all non-dominated subsets of items and verifies whether pertinent subsets fit into the container using an adapted version of our linear model. Computational experiments using large benchmark instances attest that this enumerative algorithm generally provides better solutions than the best heuristics from the literature when maximizing the number of items packed. Instances with up to 30 items are now solved to optimality, against the eight-item instance previously solved.

Keywords: Packing; Combinatorial optimization; Circular container; Cutting plane method; Parallel computing
