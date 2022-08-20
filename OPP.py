from typing import Tuple, List

import numpy as np
import gurobipy as gp
from gurobipy import GRB


class OPP:
    def __init__(self, dataset: List[Tuple],
                 radius,
                 rotation: bool = False,
                 name: str = "2D_OPP"):
        self.rotation = rotation

        if self.rotation:
            raise NotImplementedError
            # self.data = dataset + [(d[1], d[0]) for d in dataset]
        else:
            self.data = dataset

        self.R = radius
        self._name = name
        self._model = gp.Model(self._name)

        self._constr = {}

        directions = 4

        self._l = [d[0] for d in self.data]
        self._h = [d[1] for d in self.data]
        self._N = len(self.data)
        self._items = range(self._N)
        self._directions = range(directions)

        self.M = np.ones((self._N, self._N, directions)) * 2 * self.R

        self.x = self.y = self.delta = None
        self.reset_model()

    def reset_model(self):
        self._model = gp.Model(self._name)
        variables = self._add_variables()
        self.x, self.y, self.delta = variables
        self._add_constr(*variables)

        self._model.setObjective(sum(1 for _ in self._items), GRB.MAXIMIZE)

    def optimize(self):
        self._model.optimize()

    def _add_variables(self):
        x = self._model.addVars(self._N, vtype=GRB.CONTINUOUS, name="x")
        y = self._model.addVars(self._N, vtype=GRB.CONTINUOUS, name="y")
        delta = self._model.addVars(self._N, self._N, 4, vtype=GRB.BINARY,
                                    name="delta")

        return x, y, delta

    def _add_constr(self, x, y, delta):
        self._add_xy_boundaries_constr(x, y)
        self._add_no_overlap_constr(x, y, delta)
        self._add_delta_constr(delta)

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
            (x[i] == [0, 2 * self.R - self._l[i]] for i in self._items), name="10")
        self._constr["11"] = self._model.addConstrs(
            (y[i] == [0, 2 * self.R - self._h[i]] for i in self._items), name="11")

    def _add_delta_constr(self, delta):
        self._constr["19"] = self._model.addConstrs(
            (sum(delta[i, j, p] for p in self._directions) >= 1 for i in self._items
             for j in range(i + 1, self._N)), name="19"
        )


if __name__ == "__main__":
    R = 1.5
    # data = [(1, 2), (3, 1), (3, 1), (2, 1)]
    data = [(1, 2) for _ in range(3)]
    opp = OPP(dataset=data, radius=R)
    opp.optimize()
    data.append((1, 2))
    print("\n\n___________________________________________________________\n\n")
    opp = OPP(dataset=data, radius=R)
    opp.optimize()
