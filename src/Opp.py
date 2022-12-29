from itertools import combinations
from typing import List

import os
import numpy as np
import gurobipy as gp

from gurobipy import GRB

from src.Solution import Solution
from pathlib import Path

script_dir = Path(__file__).parent
license_file_path = script_dir.parent.parent.parent.parent / "gurobi952/linux64/guroby952/gurobi.lic"

os.environ['GRB_LICENSE_FILE'] = license_file_path.as_posix()
NPA = np.ndarray

"""
This module contains the implementation of the OPP algorithm.
For reference, see the paper:
    https://www.sciencedirect.com/science/article/pii/S037722172200128X?via%3Dihub

Refactoring notes:
    - Maybe a Class Problem should be implemented, 
        that contains the problem data (l, h, R...)
    - Maybe a Class Circle should be implemented, that implements geometrical 
        properties. Computing Areas, sagittas, etc. should be their responsability.
    - Maybe reduce the number of properties in the classes OPP.
    - Refactor so that add_variables returns a dictionary of variables instead of a tuple.
    - Refactor entangling Opp and Opp_rot with composition instead of inheritance.
"""


class Opp:
    def __init__(self,
                 dataset: NPA,
                 radius,
                 rotation: bool = False,
                 optimizations: List[str] | None = None,
                 name: str = "2D_OPP"):
        self.rotation = rotation

        self.data = self._handle_data_and_rotation(dataset)
        self.R = radius
        self._name = name
        self._model: gp.Model | None = None
        self.is_solved = False
        if optimizations is None:
            optimizations = []
        self.optimizations = optimizations
        self._constr = {}

        directions = 4

        self._l = np.array([d[0] for d in self.data])
        self._h = np.array([d[1] for d in self.data])
        self._N = len(self.data)
        self._items = range(self._N)
        self._items_combinations = tuple(combinations(self._items, 2))
        self._directions = range(directions)

        self.M = self._compute_M()

        self._x = self._y = self._delta = None
        self.build_model()

    @property
    def solution(self):
        """
        Return the solution of the model.
        For Opp correspond to the guroby solution.
        """
        return self.gurobi_solution

    @property
    def gurobi_solution(self) -> Solution:
        """
        Return the solution of the model obtained by the guroby solver.
        If Cpp it may contain Infeasible Items (i.e. items that are placed outside the box).

        """
        assert self.is_solved
        return Solution(self.accepted_pos, self.accepted_dims, self._accepted_values)

    @property
    def _accepted_values(self) -> NPA:
        """
        Return the accepted values of the model as a boolean vector.
        For Opp it's empty if the model is infeasible, otherwise it's the complete
        list of items.

        """
        if self.is_solved:
            return np.ones_like(self.accepted_x)
        return np.array([], dtype=bool)

    @property
    def accepted_dims(self):
        """ Return the dimensions of the accepted items."""
        return np.stack((self._l[self.accepted], self._h[self.accepted]), axis=1)

    @property
    def dims(self):
        """
        Return the dimensions of all items of the problem.
        dev: maybe should be moved to a class Problem.
        """
        return np.stack((self._l, self._h), axis=1)

    @property
    def x(self):
        """
        Return the decision variables of the x coordinates o its value if the model is solved.
        """
        if self.is_solved:
            return np.array([v.x for v in self._x.values()])
        return self._x.values()

    @property
    def y(self):
        if self.is_solved:
            return np.array([v.x for v in self._y.values()])
        return self._y.values()

    @property
    def pos(self):
        """Return the positions of all items, even not accepted."""
        return np.stack((self.x, self.y), axis=1)

    @property
    def accepted_pos(self):
        """Return the positions of accepted items."""
        return np.stack((self.accepted_x, self.accepted_y), axis=1)

    @property
    def accepted_x(self):
        """Return the x coordinates of accepted items."""
        return np.array([xi for xi in self.x[self.accepted]])

    @property
    def accepted_y(self):
        """Return the y coordinates of accepted items."""
        return np.array([yi for yi in self.y[self.accepted]])

    @property
    def accepted(self):
        """
        Return the indices of the accepted items. Opp is a feasibility problem,
        so return all items if a solution is found
        """
        if self.is_solved:
            return self._items
        return []

    def build_model(self):
        """Construct the model:
        - instatiate the underlying Gurobi model
        - add variables
        - add constraints
        - add objective function
        """
        self._model = gp.Model(self._name)
        variables = self._add_variables()
        self._x, self._y, self._delta = variables
        self._add_constr(variables)

        self._model.setObjective(sum(1 for _ in self._items), GRB.MAXIMIZE)

    def reset(self):
        """Set the model to its initial state."""
        self._model.reset()
        self.is_solved = False

    def optimize(self):
        """Solve the model."""
        self._model.optimize()
        if self._model.SolCount < 1:
            # if self._model.Status == 9:  # time limit exceed: no solution Found
            return
        self.is_solved = True

    def _add_variables(self):
        """Adding the necessary variables to the model"""
        x = self._model.addVars(self._N, vtype=GRB.CONTINUOUS, name="x")
        y = self._model.addVars(self._N, vtype=GRB.CONTINUOUS, name="y")
        delta = self._model.addVars(self._N, self._N, 4, vtype=GRB.BINARY,
                                    name="delta")

        return x, y, delta

    def _add_constr(self, variables):
        """Add all the necessary constraints to solve the model."""
        x, y, delta = variables
        self._add_xy_boundaries_constr(x, y)
        self._add_no_overlap_constr(x, y, delta)
        self._add_delta_constr(delta)

    def _compute_M(self):
        """Computes the naive big M value needed for a few constraints."""
        return np.ones((self._N, self._N, 4)) * 2 * self.R

    def _add_no_overlap_constr(self, x, y, delta):
        """
        Add the no overlap constraints that the boxes should not overlap.
        They correspond to the constraints (5) (6) (7) and (8) in the paper.
        """
        self._constr["5"] = self._model.addConstrs(
            (x[i] + self._l[i] <= x[j] + self.M[i, j, 0] * (1 - delta[i, j, 0])
             for i, j in self._items_combinations), name="5"
        )

        self._constr["6"] = self._model.addConstrs(
            (x[i] >= self._l[j] + x[j] - self.M[i, j, 1] * (1 - delta[i, j, 1])
             for i, j in self._items_combinations), name="6"
        )

        self._constr["7"] = self._model.addConstrs(
            (y[i] + self._h[i] <= y[j] + self.M[i, j, 2] * (1 - delta[i, j, 2])
             for i, j in self._items_combinations), name="7"
        )

        self._constr["8"] = self._model.addConstrs(
            (y[i] >= self._h[j] + y[j] - self.M[i, j, 3] * (1 - delta[i, j, 3])
             for i, j in self._items_combinations), name="8"
        )

    def _add_xy_boundaries_constr(self, x, y):
        """
        Add the boundaries constraints that the x and y coordinates of the items.
        This is the more permissive one, it simply says that the x and y coordinates
        should be inside the rectangle of size 2R x 2R.
        These constraints correspond to the constraints (10) and (11) in the paper.
        """
        self._constr["10"] = self._model.addConstrs(
            (x[i] == [0, 2 * self.R - self._l[i]] for i in self._items), name="10")
        self._constr["11"] = self._model.addConstrs(
            (y[i] == [0, 2 * self.R - self._h[i]] for i in self._items), name="11")

    def _add_delta_constr(self, delta):
        """Add the constaint (19) of the paper."""
        self._constr["19"] = self._model.addConstrs(
            (sum(delta[i, j, p] for p in self._directions) >= 1 for i, j in
             self._items_combinations), name="19"
        )

    def _compute_sagittas(self):
        """
        Computes the sagittas of the items in the circles.
        They are needed for VI1 (see paper (27) and (28)) and
        to compute the optimal big_M values.
        dev: It is used only in Cpp, it should be moved there or to an external class.
        """
        s_l = self.R - np.sqrt(self.R * self.R - self._l * self._l * 0.25)
        s_h = self.R - np.sqrt(self.R * self.R - self._h * self._h * 0.25)
        return s_l, s_h

    def _handle_data_and_rotation(self, dataset: NPA):
        """Return the data in the right format and handle the rotation."""
        if self.rotation:
            if type(self) is Opp:
                raise AttributeError(
                    "Cannot instantiate Opp_rot without rotation, use Opp_rot instead")
            return np.vstack((dataset, dataset[:, ::-1]))
        return dataset

    def print_solution(self):
        """Print the solution."""
        if not self.is_solved:
            print("No solution found")
            return
        print("Solution found")
        for i in self._items:
            print(f"Item {i}: x = {self._x[i].x}, y = {self._y[i].x}")


if __name__ == "__main__":
    # Toy test example
    R = 1.5
    data = np.array([(1, 2) for _ in range(3)])
    opp = Opp(dataset=data, radius=R, rotation=False)
    opp.optimize()
    data = np.array([(1, 2) for _ in range(4)])
    print("\n\n___________________________________________________________\n\n")
    opp = Opp(dataset=data, radius=R)
    opp.optimize()
