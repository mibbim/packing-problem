from typing import Literal, Tuple, List

import numpy as np
from itertools import combinations, chain
from gurobipy import GRB

from Opp import Opp


def powerset(iterable, r):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return list(chain.from_iterable(map(list, combinations(xs, n)) for n in range(r + 1)))


class Rpp(Opp):
    def __init__(self,
                 dataset: List[Tuple],
                 values: Literal["count", "volume"],
                 radius: float,
                 rotation: bool = False,
                 optimizations: List | None = None,
                 name: str = "2D_Rpp"):
        self.rotation = rotation
        self.data = self._handle_data_and_rotation(dataset)
        if values == "count":
            self._v = [1 for _ in self.data]
        elif values == "volume":
            self._v = [l * h for (l, h) in self.data]
        else:
            raise NotImplementedError("No value recognized")

        self._a = self._z = self._x = self._y = self._delta = None

        super().__init__(dataset=self.data,
                         radius=radius,
                         rotation=self.rotation,
                         optimizations=optimizations,
                         name=name)

    @property
    def accepted(self):
        assert self.is_solved
        return np.array([i for i in self._items if round(self._a[i].x)])

    def build_model(self):
        variables = self._add_variables()
        self._a, self._z, self._x, self._y, self._delta = variables
        self._add_constr(variables)
        self._model.setObjective(sum(self._a[i] * self._v[i] for i in self._items),
                                 GRB.MAXIMIZE)
        return variables

    def display(self, plot=True, show=True):
        print(f"\n\n___________ Solution of problem: {self._name} ___________")
        if self.rotation:
            for i in range(self._N // 2):
                if i in self.accepted:
                    print(f"Accepted item {i}, at position ({self._x[i].x},{self._y[i].x})")
                if i + self._N // 2 in self.accepted:
                    print(f"Accepted item {i} ({i + self._N // 2}), at position "
                          f"({self._x[i + self._N // 2].x},{self._y[i + self._N // 2].x}) rotated")
        else:
            for i in self.accepted:
                print(f"Accepted item {i}, at position ({self._x[i].x},{self._y[i].x})")

        if plot:
            import matplotlib.pyplot as plt

            ax = plt.gca()
            ax.cla()
            for i, (p, d) in enumerate(zip(self.pos, self.dims)):
                if i in self.accepted:
                    r = plt.Rectangle(p, d[0], d[1], linewidth=1, edgecolor='g', facecolor='none')
                    ax.annotate(str(i), xy=p + 0.5 * d)
                    ax.add_patch(r)
            c = plt.Circle((self.R, self.R), self.R, alpha=0.5)
            ax.set_xlim((0, 2 * self.R))
            ax.set_ylim((0, 2 * self.R))
            ax.add_patch(c)
            # plt.legend()
            if show:
                plt.show()
            else:
                print(f"__________________________________________________\n\n")
                return ax

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
        if "delta" in self.optimizizations:
            self._add_delta_bound_constr(z, delta)
        if "area" in self.optimizizations:
            self._add_area_constraint(a)
        if "feasible_subsets" in self.optimizizations:
            self._add_feasible_subsets(a)
        if self.rotation:
            self._add_rotation_constr(a)

    def _add_z_definitions_constr(self, a, z):
        self._constr["2_1"] = self._model.addConstrs(
            (a[i] + a[j] - 1 <= z[i, j] for i, j in self._items_combinations),
            name="2a"
        )
        self._constr["2_2"] = self._model.addConstrs(
            (z[i, j] <= a[i] for i, j in self._items_combinations),
            "2b"
        )
        self._constr["3_2"] = self._model.addConstrs(
            (z[i, j] <= a[j] for i, j in self._items_combinations),
            "3b"
        )
        # Theoretical equivalent
        # self._constr["z_def"] = self._model.addConstrs(

    #     self._z[i, j] == and_(self._a[i], self._a[j]) for i, j in self._items_combinations)

    def _add_delta_bound_constr(self, z, delta):
        """just for speedup"""
        self._constr["4_1"] = self._model.addConstrs(
            (z[i, j] <= sum(delta[i, j, p] for p in self._directions) for i, j in
             self._items_combinations), "4a"
        )
        self._constr["4_2"] = self._model.addConstrs(
            (sum(delta[i, j, p] for p in self._directions) <= 2 * z[i, j] for i, j in
             self._items_combinations), "4b"
        )

    def _add_area_constraint(self, a):
        acceptable_item_num = self._get_max_acceptable_item_num()
        self._model.addConstr(sum(a[i] for i in self._items) <= acceptable_item_num, name="area")

    def _get_max_acceptable_item_num(self):
        total_area = self._total_area()
        items_areas = self._get_items_sorted_areas()
        areas_sum = np.cumsum(items_areas)
        return np.sum(areas_sum < total_area)

    def _get_items_sorted_areas(self):
        return np.sort(self._l * self._h)

    def _add_feasible_subsets(self, a):
        max_k = np.ceil(np.sqrt(self.dims.shape[0])).astype(int)

        subsets = np.array(powerset(self._items, max_k), dtype=object)
        areas = self._l * self._h
        subsets_total_areas = np.array([areas[s].sum() for s in subsets])
        infeasible = subsets_total_areas > self._total_area()
        # infeasible = [1, 3, 4, 300, -1]
        for s in subsets[infeasible]:
            self._model.addConstr(
                (sum(a[i] for i in self._items if i in s) <= len(s) - 1),
                name="set_feasibility"
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
