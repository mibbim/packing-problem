from typing import Tuple, List, Literal

import matplotlib.pyplot as plt
import numpy as np
from time import time

from src.Opp import Opp
from src.RPP import Rpp
from src.circular_container_cuts import CircularContainerCuts

cut_tol = 1e-4


class Cpp(Rpp):
    def __init__(self,
                 dataset: List[Tuple],
                 values: Literal["count", "volume"],
                 radius: float,
                 rotation: bool = False,
                 optimizations: List | None = None,
                 name: str = "2D_Cpp"):
        feasible_data = self._get_feasible_items(radius, dataset)
        Rpp.__init__(self, feasible_data, values, radius, rotation, optimizations, name)
        self._values = values
        self.ccc = None
        self._prev_as = np.empty((0, 2))

    @property
    def area(self):
        assert self.is_solved
        return self.accepted_dims.prod(axis=1).sum()

    @property
    def count(self):
        assert self.is_solved
        return self.accepted.shape[0]

    @property
    def obj_val(self):
        assert self.is_solved
        return self._model.getObjective().getValue()


    def _compute_M(self):
        M = super(Cpp, self)._compute_M()
        if "big_M" in self.optimizizations:
            return self._optimize_M(M)
        return M

    def _optimize_M(self, M):
        s_l, s_h = self._compute_sagittas()
        s_h_sum = np.add.outer(s_h, s_h)
        s_l_sum = np.add.outer(s_l, s_l)
        M[:, :, 0] -= s_h_sum
        M[:, :, 1] -= s_h_sum
        M[:, :, 2] -= s_l_sum
        M[:, :, 3] -= s_l_sum
        return M

    @staticmethod
    def _get_feasible_items(r, dataset):
        """Needs R"""
        return dataset[np.linalg.norm(dataset, axis=1) <= 2 * r]

    def _add_constr(self, variables):
        a, z, x, y, delta = variables
        super(Cpp, self)._add_constr(variables)
        if "infeasible_pairs" in self.optimizizations:
            self._add_infeasible_pairs_opt(a)
        if "symmetry" in self.optimizizations:
            self._break_simmetry(a, x, y)

    def _break_simmetry(self, a, x, y):
        self._model.addConstrs(
            (x[i] + 0.5 * self._l[i] <= self.R * (1 + sum(a[j] for j in range(i)))
             for i in self._items))
        self._model.addConstrs(
            (y[i] + 0.5 * self._h[i] <= self.R * (1 + sum(a[j] for j in range(i)))
             for i in self._items))

    def compute_tangent_angular_coefficient(self, a):
        a_minus_r = a - self.R
        return - a_minus_r[:, 0] / a_minus_r[:, 1]

    def add_tangent_plane_cuts(self):
        ccc = self.ccc
        a = ccc.get_intersection_point()
        m = self.compute_tangent_angular_coefficient(a)
        accepted = self.accepted
        self.reset()
        add_cuts_methods = [self._add_cuts_s1,
                            self._add_cuts_s2,
                            self._add_cuts_s3,
                            self._add_cuts_s4]

        if "all_tangent" in self.optimizizations or "symmetric_tangent" in self.optimizizations:
            for i, add_cut_method in enumerate(add_cuts_methods):
                add_cut_method(m[ccc.s[i]], self.dims,
                               a[ccc.s[i]], self.pos)
            cuts_added = ccc.s.sum() * self.dims.shape[0]
        else:
            for i, add_cut_method in enumerate(add_cuts_methods):
                add_cut_method(m[ccc.s[i]], ccc.accepted_dims[ccc.s[i]],
                               a[ccc.s[i]], self.pos[accepted][ccc.s[i]])
            cuts_added = ccc.s.sum()
        print(f"Adding {cuts_added} cuts")
        return a[ccc.s.sum(axis=0).astype(bool)]

    def _add_cuts_s1(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi + hi <= ya + ma * (xi + li - xa)
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s1"
        )

    def _add_cuts_s2(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi + hi <= ya + ma * (xi - xa)
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s2"
        )

    def _add_cuts_s3(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi >= ya + ma * (xi - xa)
             for (xi, yi) in pos
             for ((xa, ya), ma) in zip(a, m)),
            name="s3"
        )

    def _add_cuts_s4(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi >= ya + ma * (xi + li - xa)
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s4"
        )

    def display(self, plot=True, show=True, title: str | None = None):
        if self.ccc is None:
            super(Cpp, self).display(plot, show)
            return
        if plot:
            ax = super().display(plot, show=False)
            xx = np.linspace(0, self.R * 2, 100)
            points = self.ccc.get_intersection_point()[self.ccc.s.sum(axis=0).astype(bool)]
            a_s = points
            if "all_tangent" in self.optimizizations:
                a_s = self._prev_as
            m = self.compute_tangent_angular_coefficient(a_s)
            for (xa, ya), ma in zip(a_s, m):
                ax.plot(xx, ya + (xx - xa) * ma)
            ax.scatter(points.T[0], points.T[1])
        if show:
            plt.title(title)
            plt.show()

    def optimize(self, max_iteration: int = 10, display_each: int = 2, time_limit: int = np.inf):
        if "initial_objective_bound" in self.optimizizations:
            self._model.Params.BestObjStop = self._get_initial_objective_bound()

        if "initial_cutoff" in self.optimizizations:
            self._model.Params.Cutoff = self._get_initial_cutoff()

        self._model.Params.TimeLimit = time_limit
        start = time()
        Rpp.optimize(self)
        elapsed = time() - start

        self.display(title="0")
        self.ccc = CircularContainerCuts(self)
        prev_feasible_obj = min(self.accepted_values)
        for it in range(1, max_iteration):
            prev_obj_val = self._model.objVal
            prev_feasible_obj = max(prev_feasible_obj, self.ccc.feasible_obj)

            print(f"Iteration: {it}_________________________-")

            if self.ccc.acceptable:
                print(f"Solution found at iteration {it}******************************************")
                self.display(title=f"Solution Found: {it}")
                break

            if elapsed < 0:
                # TODO implement history and get best solution
                print(f"Best found at iteration {it}******************************************")

                break

            a = self.add_tangent_plane_cuts()
            self._prev_as = np.vstack((self._prev_as, a))

            if "objective_bound" in self.optimizizations:
                self._model.Params.BestObjStop = prev_obj_val + 1e-4
                print(f"Objective bound: {prev_obj_val}")

            if "cutoff" in self.optimizizations:
                self._model.Params.Cutoff = prev_feasible_obj - 1e-4
                print(f"Cutoff value: {prev_feasible_obj}")

            self._model.Params.TimeLimit = max(time_limit - elapsed, 0)
            super().optimize()
            elapsed = time() - start

            if it % display_each == 0:
                self.display(title=f"{it}")
            self.ccc = CircularContainerCuts(self)
        print()
        return elapsed

    def _add_xy_boundaries_constr(self, x, y):
        if "sagitta" in self.optimizizations:
            self._add_xy_sagitta_boundaries(x, y)
            return
        Opp._add_xy_boundaries_constr(self, x, y)

    def _add_xy_sagitta_boundaries(self, x, y):
        s_l, s_h = self._compute_sagittas()
        self._constr["10"] = self._model.addConstrs(
            (x[i] == [s_h[i], 2 * self.R - self._l[i] - s_h[i]] for i in self._items), name="10")
        self._constr["11"] = self._model.addConstrs(
            (y[i] == [s_l[i], 2 * self.R - self._h[i] - s_l[i]] for i in self._items), name="11")

    def _add_infeasible_pairs_opt(self, a):
        if "big_M" in self.optimizizations:
            bound = self.M
        else:
            bound = self._optimize_M(self.M.copy())
        bound = bound[:, :, (0, 3)]
        l_sum = np.add.outer(self._l, self._l)
        h_sum = np.add.outer(self._h, self._h)
        infeasible = (l_sum >= bound[:, :, 0]) & (h_sum >= bound[:, :, 1])
        self._model.addConstrs(
            (a[i] + a[j] <= 1 for i, j in self._items_combinations if infeasible[i, j]),
            name="infeasible_pairs"
        )

    def _total_area(self):
        return self.R * self.R * np.pi

    def _get_initial_objective_bound(self):
        if self._values == "volume":
            return self._total_area()
        return self._get_max_acceptable_item_num()

    def _get_initial_cutoff(self):
        return max(self.values)


if __name__ == "__main__":
    from datetime import datetime

    N = 10  # 20
    rho = 0.33

    rng = np.random.default_rng(42)
    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)

    start = datetime.now()
    print(start)
    opts = [
        "big_M",  #
        "delta",  #
        "sagitta",  #
        "area",  #
        "feasible_subsets",  #
        "infeasible_pairs",  #
        "symmetry",  #
        "all_tangent",
        "objective_bound",
        "cutoff",
        "initial_objective_bound",
        "initial_cutoff"

        # "initial_tangent"
        # "symmetric_tangent",
    ]
    print(f"Using optimizations {opts}, {N=}, {circle_area=}, {R=}")
    cpp = Cpp(dataset=data, values="volume", radius=R, optimizations=opts)
    cpp.optimize(10, 1, time_limit=500)
    stop = datetime.now()
    print(start, stop, stop - start)
