from typing import Literal, Tuple, List

import gurobipy as gp
from gurobipy import GRB

from OPP import OPP


class Rpp(OPP):
    def __init__(self, dataset: List[Tuple],
                 values: Literal["count", "volume"],
                 radius,
                 rotation: bool = False,
                 name: str = "2D_Rpp"):
        self.rotation = rotation

        if self.rotation:
            self.data = dataset + [(d[1], d[0]) for d in dataset]
        else:
            self.data = dataset

        if values == "count":
            self._v = [1 for _ in self.data]
        elif values == "volume":
            self._v = [l * h for (l, h) in self.data]
        else:
            raise NotImplementedError("No value recognized")

        self.a = self.z = self.x = self.y = self.delta = None

        super().__init__(dataset=self.data, radius=radius, rotation=self.rotation, name=name)
        # self.rotation = rotation

    def reset_model(self):
        self._model = gp.Model(self._name)
        variables = self._add_variables()
        self.a, self.z, self.x, self.y, self.delta = variables
        self._add_constr(variables)
        self._model.setObjective(sum(self.a[i] * self._v[i] for i in self._items), GRB.MAXIMIZE)
        return variables

    def display(self):
        print(f"\n\n___________ Solution of problem: {self._name} ___________")
        accepted = [i for i in self._items if self.a[i].x > 1e-6]
        if self.rotation:
            for i in range(self._N // 2):
                if i in accepted:
                    print(f"Accepted item {i}, at position ({self.x[i].x},{self.y[i].x})")
                if i + self._N // 2 in accepted:
                    print(f"Accepted item {i} ({i + self._N // 2}), at position "
                          f"({self.x[i + self._N // 2].x},{self.y[i + self._N // 2].x}) rotated")
        else:
            for i in accepted:
                print(f"Accepted item {i}, at position ({self.x[i].x},{self.y[i].x})")

        print("Z:")
        print(f"   {[j for j in accepted[1:]]}")
        for i in accepted[:-1]:
            print(f"{i}:{[self.z[i, j].x for j in accepted[1:]]}")
        # i = 3
        # j = 6
        # print(f"Delta {i}, {j}:")
        # print(f"{[self.delta[i, j, p].x for p in range(4)]}")
        print(f"__________________________________________________\n\n")

    def _add_variables(self):
        a = self._model.addVars(self._N, vtype=GRB.BINARY, name="a")  # acceptance of box i
        z = self._model.addVars(self._N, self._N, vtype=GRB.BINARY, name="z")
        x, y, delta = super()._add_variables()

        return a, z, x, y, delta

    def _add_constr(self, variables):
        a, z, x, y, delta = variables
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

    def _add_rotation_constr(self, a):
        self._constr["rotation"] = self._model.addConstrs(
            a[i] + a[i + self._N // 2] <= 1 for i in range(self._N // 2)
        )


if __name__ == "__main__":
    R = 1.5
    # data = [(1, 2), (3, 1), (3, 1), (2, 1)]
    data = [(1, 2) for _ in range(4)]

    rpp = Rpp(dataset=data, values="volume", radius=R)
    rpp.optimize()
    rpp.display()

    rot = Rpp(dataset=data, values="volume", radius=R, rotation=True)
    rot.optimize()
    rot.display()
    print()
