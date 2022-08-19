from typing import Literal, Tuple, List

import numpy as np
import gurobipy as gp
from gurobipy import GRB


class RPP:
    def __init__(self, dataset: List[Tuple],
                 values: Literal["count", "volume"],
                 radius,
                 rotation: bool = False,
                 name: str = "2D_Rpp"):
        """
        Select the rectangles to be packed in a rectangular box tha maximizes the value
        """
        self.rotation = rotation

        if self.rotation:
            self.data = dataset + [(d[1], d[0]) for d in dataset]
        self.data = dataset
        self.R = radius
        self._name = name
        self._model = gp.Model(self._name)

        self._constr = {}

        if values == "count":
            self._v = [1 for (l, h) in self.data]
        elif values == "volume":
            self._v = [l * h for (l, h) in self.data]
        else:
            raise NotImplementedError("No value recognized")

        directions = 4

        self._l = [d[0] for d in self.data]
        self._h = [d[1] for d in self.data]
        self._N = len(self.data)
        self._items = range(self._N)
        self._directions = range(directions)

        self.M = np.ones((self._N, self._N, directions)) * 2 * R

        self.a, self.z, self.x, self.y, self.delta = self.reset_model()
        self._model.setObjective(sum(self.a[i] * self._v[i] for i in self._items), GRB.MAXIMIZE)
        # for CPP
        # self._s_l = np.zeros(self._N)
        # self._s_h = np.zeros(self._N)

    def reset_model(self):
        self._model = gp.Model(self._name)
        variables = self._add_variables()
        self._add_constr(*variables)
        a = variables[0]
        self._model.setObjective(sum(a[i] * self._v[i] for i in self._items), GRB.MAXIMIZE)
        return variables

    def optimize(self):
        self._model.optimize()

    def _add_variables(self):
        a = self._model.addVars(self._N, vtype=GRB.BINARY, name="a")  # acceptance of box i
        z = self._model.addVars(self._N, self._N, vtype=GRB.BINARY, name="z")
        x = self._model.addVars(self._N, vtype=GRB.CONTINUOUS, name="x")
        y = self._model.addVars(self._N, vtype=GRB.CONTINUOUS, name="y")
        delta = self._model.addVars(self._N, self._N, 4, vtype=GRB.BINARY,
                                    name="delta")
        return a, z, x, y, delta

    def _add_constr(self, a, z, x, y, delta):
        self._add_z_definitions_constr(a, z)
        self._add_no_overlap_constr(x, y, delta)
        self._add_xy_boundaries_constr(x, y)
        self._add_delta_bound_constr(z, delta)
        if self.rotation:
            self._add_rotation_constr(a)

    def _add_z_definitions_constr(self, a, z):
        self._constr["2_1"] = self._model.addConstrs(
            (a[i] + a[j] - 1 <= z[i, j] for i in self._items for j in
             range(i + 1, self._N)), "2a"
        )
        self._constr["2_2"] = self._model.addConstrs(
            (z[i, j] <= a[i] for i in self._items for j in range(i + 1, self._N)),
            "2b"
        )
        self._constr["3_2"] = self._model.addConstrs(
            (z[i, j] <= a[j] for i in self._items for j in range(i + 1, self._N)),
            "3b"
        )
        # Theoretical equivalent
        # self._constr["z_def"] = self._model.addConstrs(
        #     self.z[i, j] == and_(self.a[i], self.a[j]) for i in self._items for j in
        #     range(i + 1, self._N))

    def _add_delta_bound_constr(self, z, delta):
        """just for speedup"""
        self._constr["4_1"] = self._model.addConstrs(
            (z[i, j] <= sum(delta[i, j, p] for p in self._directions) for i in self._items
             for j in range(i + 1, self._N)), "4a"
        )
        self._constr["4_2"] = self._model.addConstrs(
            (sum(delta[i, j, p] for p in self._directions) <= 2 * z[i, j] for i in
             self._items for j in range(i + 1, self._N)), "4b"
        )

    def _add_no_overlap_constr(self, x, y, delta):
        self._constr["5"] = self._model.addConstrs(
            (x[i] + self._l[i] <= x[j] + self.M[i, j, 0] * (1 - delta[i, j, 0])
             for i in self._items for j in range(i + 1, self._N)), name="5"
        )

        self._constr["6"] = self._model.addConstrs(
            (x[i] >= self._l[j] + x[j] - self.M[i, j, 1] * (1 - delta[i, j, 1])
             for i in self._items for j in range(i + 1, self._N)), name="6"
        )

        self._constr["7"] = self._model.addConstrs(
            (y[i] + self._h[i] <= y[j] + self.M[i, j, 2] * (1 - delta[i, j, 2])
             for i in self._items for j in range(i + 1, self._N)), name="7"
        )

        self._constr["8"] = self._model.addConstrs(
            (y[i] >= self._h[j] + y[j] - self.M[i, j, 3] * (1 - delta[i, j, 3])
             for i in self._items for j in range(i + 1, self._N)), name="8"
        )

    def _add_xy_boundaries_constr(self, x, y):
        self._constr["10"] = self._model.addConstrs(
            (x[i] == [0, 2 * self.R - self._l[i]] for i in self._items), name="11")
        self._constr["11"] = self._model.addConstrs(
            (y[i] == [0, 2 * self.R - self._h[i]] for i in self._items), name="11")

    def _add_rotation_constr(self, a):
        self._constr["rotation"] = self._model.addConstrs(
            a[i] + a[i + self._N // 2] <= 1 for i in range(self._N // 2)
        )


if __name__ == "__main__":
    R = 1.5
    data = [(1, 2), (3, 1), (3, 1), (2, 1)]

    rpp = RPP(dataset=data, values="volume", radius=R)
    rpp.optimize()

    print()
