from typing import Literal, List

import numpy as np
from itertools import combinations, chain
from gurobipy import GRB, tupledict
from matplotlib import pyplot as plt

from src.Opp import Opp
from src.Solution import add_solution_rectangles, Solution

DecisionVariable = tupledict


def create_new_sqauared_ax(l: float) -> plt.Axes:
    fig, ax = plt.subplots()
    ax.set_xlim(0, l)
    ax.set_ylim(0, l)
    return ax


def powerset(iterable, r):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return list(chain.from_iterable(map(list, combinations(xs, n)) for n in range(r + 1)))


class Rpp(Opp):
    def __init__(self,
                 dataset: np.ndarray,
                 values: Literal["count", "volume"],
                 radius: float,
                 rotation: bool = False,
                 optimizations: List | None = None,
                 name: str = "2D_Rpp"):
        self.rotation = rotation
        self.data = self._handle_data_and_rotation(dataset)
        if values == "count":
            self._v = np.ones(self.data.shape[0])  # [1 for _ in self.data]
        elif values == "volume":
            self._v = np.prod(self.data, axis=1)
        else:
            raise NotImplementedError("No value recognized")

        self._a = self._z = self._x = self._y = self._delta = None

        super().__init__(dataset=dataset,
                         radius=radius,
                         rotation=self.rotation,
                         optimizations=optimizations,
                         name=name)

    @property
    def solution(self):
        return Solution(
            self.accepted_pos,
            self.accepted_dims,
            self._accepted_values,
        )

    @property
    def accepted(self):
        assert self.is_solved
        return np.array([i for i in self._items if round(self._a[i].x)])

    @property
    def values(self):
        return self._v

    @property
    def _accepted_values(self):
        return self.values[self.accepted]

    def build_model(self):
        variables = self._add_variables()
        self._a, self._z, self._x, self._y, self._delta = variables
        self._add_constr(variables)
        self._model.setObjective(sum(self._a[i] * self._v[i] for i in self._items),
                                 GRB.MAXIMIZE)
        return variables

    def display(self, title: str = "", plot: bool = True, show: bool = True):
        if plot:
            ax = create_new_sqauared_ax(2 * self.R)

            self.gurobi_solution.display(self.R, ax=ax, color="r")
            add_solution_rectangles(ax=ax, solution=self.solution, color="g")

            if show:
                ax.set_title(title)
                plt.show()

    def _add_variables(self):
        a = self._model.addVars(self._N, vtype=GRB.BINARY, name="a")  # acceptance of box i
        z = self._model.addVars(self._N, self._N, vtype=GRB.BINARY, name="z")
        x, y, delta = super()._add_variables()
        return a, z, x, y, delta

    def _add_constr(self, variables):
        a: DecisionVariable  # decision variable: acceptance of box i
        a, z, x, y, delta = variables
        self._add_z_definitions_constr(a, z)
        self._add_no_overlap_constr(x, y, delta)
        self._add_xy_boundaries_constr(x, y)
        self._add_delta_bound_constr(z, delta)
        if "area" in self.optimizizations:
            self._add_area_constraint(a)
        if "feasible_subsets" in self.optimizizations:
            self._add_feasible_subsets(a)
        if self.rotation:
            self._add_rotation_constr(a)

    def _add_z_definitions_constr(self, a, z):
        """
        Add the constraints that define the z variables.
        Correspond to (2) and (3) in the paper.
        """
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
        """
        Add an upper bound on the number of the accepted items based on
        the sum of the areas of smallest accepted items.
        Refer to section 3.5.2 of the paper.
        Correspond to inequality (31) in the paper also referred as VI2.
        """
        acceptable_item_num = self._get_max_acceptable_item_num()
        self._model.addConstr(sum(a[i] for i in self._items) <= acceptable_item_num, name="area")

    def _get_max_acceptable_item_num(self):
        """Computes the upper bound on the number of the accepted items based on
        the sum of the areas of smallest accepted items."""
        total_area = self._total_area()
        items_areas = self._get_items_sorted_areas()
        areas_sum = np.cumsum(items_areas)
        return np.sum(areas_sum < total_area)

    def _get_items_sorted_areas(self):
        return np.sort(self._l * self._h)

    def _add_feasible_subsets(self, acceptance_decision_variables):
        max_k = np.ceil(np.sqrt(self.dims.shape[0])).astype(int)

        subsets = np.array(powerset(self._items, max_k), dtype=object)
        areas = self._l * self._h
        subsets_total_areas = np.array([areas[s].sum() for s in subsets])
        infeasible = subsets_total_areas > self._total_area()
        for s in subsets[infeasible]:
            self._model.addConstr(
                (sum(acceptance_decision_variables[i] for i in self._items if i in s) <= len(
                    s) - 1), name="set_feasibility"
            )

    def _add_rotation_constr(self, a):
        self._constr["rotation"] = self._model.addConstrs(
            a[i] + a[i + self._N // 2] <= 1 for i in range(self._N // 2)
        )

    def _total_area(self):
        return self.R * self.R


if __name__ == "__main__":
    R = 1.5
    # data = [(1, 2), (3, 1), (3, 1), (2, 1)]
    data = np.array([(1, 2) for _ in range(4)])

    opts = []

    rpp = Rpp(data, values="volume", radius=R, optimizations=opts, rotation=False)
    rpp.optimize()
    rpp.display()

    rot = Rpp(dataset=data, values="volume", radius=R, rotation=True)
    rot.optimize()
    rot.display()
    print()
