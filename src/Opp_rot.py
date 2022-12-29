from __future__ import annotations

from typing import List

from src.Opp import Opp, NPA

from gurobipy import GRB
import gurobipy as gp


class Opp_rot(Opp):
    """ Orthogonal packing problem implementation that allows rotation."""

    def __init__(self,
                 dataset: NPA,
                 radius: float,
                 rotation: bool = True,
                 optimizations: List[str] | None = None,
                 name: str = "2D_OPP_R"):
        self._r: gp.Var | None = None  # rotation variable
        super().__init__(dataset, radius, rotation=rotation, optimizations=optimizations, name=name)

    def _handle_data_and_rotation(self, dataset: NPA):
        # if not self.rotation:
        #     raise AttributeError("Cannot instantiate Opp_rot without rotation, use Opp instead")

        return dataset

    def _add_variables(self):
        """Adding the necessary variables to the model"""
        if not self.rotation:
            return Opp._add_variables(self)

        x, y, delta = super()._add_variables()
        r = self._model.addVars(self._N, vtype=GRB.BINARY, name="r")
        return x, y, delta, r

    def _add_xy_boundaries_constr(self, x, y):
        """
        Add the boundaries constraints that the x and y coordinates of the items.
        These are equivalent to the one in the non-rotated case, but have some modifications.
        These constraints correspond to the constraints (16) and (16) in the paper.
        """
        if not self.rotation:
            Opp._add_xy_boundaries_constr(self, x, y)
            return

        r = self._r
        l, h = self._l, self._h
        x_uppers = [(1 - r[i]) * (2 * self.R - l[i]) + r[i] * (2 * self.R - h[i]) for i
                    in self._items]
        y_uppers = [(1 - r[i]) * (2 * self.R - h[i]) + r[i] * (2 * self.R - l[i]) for i
                    in self._items]
        self._constr["16_1"] = self._model.addConstrs(
            (x[i] >= 0 for i in self._items), name="16_1")

        self._constr["16_2"] = self._model.addConstrs(
            (x[i] <= x_uppers[i] for i in self._items), name="16_2")

        self._constr["17_1"] = self._model.addConstrs(
            (y[i] >= 0 for i in self._items), name="17_1")

        self._constr["17_2"] = self._model.addConstrs(
            (y[i] <= y_uppers[i] for i in self._items), name="17_2")

    def _add_no_overlap_constr(self, x, y, delta):
        """
        Add the no overlap constraints to the model.
        Correspond to the constraints (12), (13), (14) and (15) in the paper.
        """
        if not self.rotation:
            Opp._add_no_overlap_constr(self, x, y, delta)
            return

        r = self._r
        l = self._l
        h = self._h
        self._constr["12"] = self._model.addConstrs(
            (x[i] + (1 - r[i]) * l[i] + r[i] * h[i] <= x[j] + self.M[i, j, 0] * (1 - delta[i, j, 0])
             for i, j in self._items_combinations), name="12"
        )
        self._constr["13"] = self._model.addConstrs(
            (x[i] >= x[j] + (1 - r[j]) * l[j] + r[j] * h[j] - self.M[i, j, 1] * (1 - delta[i, j, 1])
             for i, j in self._items_combinations), name="13"
        )

        self._constr["14"] = self._model.addConstrs(
            (y[i] + (1 - r[i]) * h[i] + r[i] * l[i] <= y[j] + self.M[i, j, 2] * (1 - delta[i, j, 2])
             for i, j in self._items_combinations), name="14"
        )

        self._constr["15"] = self._model.addConstrs(
            (y[i] >= y[j] + (1 - r[j]) * h[j] + r[j] * l[j] - self.M[i, j, 3] * (1 - delta[i, j, 3])
             for i, j in self._items_combinations), name="15"
        )

    def _add_constr(self, variables):
        """Adding the necessary constraints to the model"""
        if not self.rotation:
            Opp._add_constr(self, variables)
            return
        x, y, delta, r = variables
        self._add_xy_boundaries_constr(x, y)
        self._add_no_overlap_constr(x, y, delta)
        self._add_delta_constr(delta)

    def build_model(self):
        """Construct the model:
        - instatiate the underlying Gurobi model
        - add variables
        - add constraints
        - add objective function
        """
        if not self.rotation:
            Opp.build_model(self)
            return
        self._model = gp.Model(self._name)
        variables = self._add_variables()
        self._x, self._y, self._delta, self._r = variables
        self._add_constr(variables)
        self._model.setObjective(sum(1 for _ in self._items), GRB.MAXIMIZE)

    def print_solution(self):
        """Print the solution of the model"""
        super().print_solution()
        if self.rotation:
            for i in self._items:
                print(f"Rotation: {self._r[i].x}")


if __name__ == "__main__":
    import numpy as np

    R = 1.5
    data = np.array([(1, 2) for _ in range(3)])
    # 1. Create a new instance of the class
    opp_rot = Opp_rot(dataset=data, radius=R, rotation=False)

    # 2. Call the method
    opp_rot.optimize()

    opp_rot.print_solution()
