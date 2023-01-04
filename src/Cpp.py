from typing import Tuple, List, Literal

import matplotlib.pyplot as plt
import numpy as np
from time import time

from src.Opp import Opp
from src.RPP import Rpp, create_new_sqauared_ax
from src.Solution import BestSolution, Solution, add_solution_rectangles
from src.circular_container_cuts import SolutionProcesser

cut_tol = 0


def compute_tangent_angular_coefficient(R: float, intersection_points: np.ndarray) -> np.ndarray:
    a_minus_r = intersection_points - R
    return - a_minus_r[:, 0] / a_minus_r[:, 1]


def draw_tanget_lines(ax: plt.Axes, R: float,
                      intersection_points: np.ndarray,
                      angular_coefficients: np.ndarray) -> None:
    xx = np.linspace(0, 2 * R, 100)
    for (xp, yp), m in zip(intersection_points, angular_coefficients):
        yy = m * (xx - xp) + yp
        ax.plot(xx, yy, color="red", alpha=0.5)


def draw_intersection_points(ax: plt.Axes, intersection_points: np.ndarray) -> None:
    ax.scatter(intersection_points[:, 0], intersection_points[:, 1])


def draw_cuts(ax: plt.Axes, R: float, intersection_points: np.ndarray) -> None:
    m = compute_tangent_angular_coefficient(R, intersection_points)
    draw_tanget_lines(ax, R, intersection_points, m)
    draw_intersection_points(ax, intersection_points)


class Cpp(Rpp):
    """Implementation of Circular Packing Problem."""

    def __init__(self,
                 dataset: List[Tuple],
                 values: Literal["count", "volume"],
                 radius: float,
                 rotation: bool = False,
                 optimizations: List | None = None,
                 name: str = "2D_Cpp",
                 rotate_with_duplicates: bool = True,
                 ):
        if self.__class__ is Cpp:
            if rotation and not rotate_with_duplicates:
                raise ValueError("Rotation is implemented only with duplicates")
        feasible_data = self._get_feasible_items(radius, dataset)
        self._duplicate = rotate_with_duplicates
        super().__init__(feasible_data, values, radius, rotation, optimizations, name,
                         rotate_with_duplicates=rotate_with_duplicates)
        self._values = values
        self.ccc = None
        self._prev_as = np.empty((0, 2))
        self._best_sol = BestSolution()
        self.history = []
        self.optimal_solution_found: bool = False

    @property
    def solution(self) -> Solution:
        return self._best_sol

    @property
    def count(self):
        assert self.is_solved
        return self.accepted.shape[0]

    def _compute_M(self):
        M = super()._compute_M()
        if "big_M" in self.optimizations:
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
    def _get_feasible_items(radius, dataset):
        """Needs R"""
        return dataset[np.linalg.norm(dataset, axis=1) <= 2 * radius - cut_tol]

    def _add_constr(self, variables):
        a, z, x, y, delta = variables
        super()._add_constr(variables)
        if "infeasible_pairs" in self.optimizations:
            self._add_infeasible_pairs_opt(a)
        if "symmetry" in self.optimizations:
            self._break_simmetry(a, x, y)

    def _break_simmetry(self, a, x, y):
        self._model.addConstrs(
            (x[i] + 0.5 * self._l[i] <= self.R * (1 + sum(a[j] for j in range(i)))
             for i in self._items))
        self._model.addConstrs(
            (y[i] + 0.5 * self._h[i] <= self.R * (1 + sum(a[j] for j in range(i)))
             for i in self._items))

    def add_tangent_plane_cuts(self):
        ccc = self.ccc
        a = ccc.get_intersection_point()
        m = compute_tangent_angular_coefficient(self.R, a)
        accepted = self.accepted
        self.reset()
        add_cuts_methods = [self._add_cuts_s1,
                            self._add_cuts_s2,
                            self._add_cuts_s3,
                            self._add_cuts_s4]

        if "all_tangent" in self.optimizations or "symmetric_tangent" in self.optimizations:
            for i, add_cut_method in enumerate(add_cuts_methods):
                add_cut_method(m[ccc.s[i]], self.dims,
                               a[ccc.s[i]], self.pos)
            cuts_added = ccc.s.sum() * self.dims.shape[0]
            # if self.rotation:  # and not self._duplicate: it is equivalent
            # add_cut_method(m[ccc.s[i]], self.dims[:, ::-1],
            #                a[ccc.s[i]], self.pos)
            # cuts_added += ccc.s.sum() * self.dims.shape[0]
        else:
            for i, add_cut_method in enumerate(add_cuts_methods):
                add_cut_method(m[ccc.s[i]], ccc.accepted_dims[ccc.s[i]],
                               a[ccc.s[i]], self.pos[accepted][ccc.s[i]])
            cuts_added = ccc.s.sum()
            # if self.rotation:  # and not self._duplicate: its equivalent
            #     add_cut_method(m[ccc.s[i]], ccc.accepted_dims[ccc.s[i]][:, ::-1],
            #                    a[ccc.s[i]], self.pos[accepted][ccc.s[i]])
            #     cuts_added += ccc.s.sum()
        print(f"Adding {cuts_added} cuts")
        return a[ccc.s.sum(axis=0).astype(bool)]

    def _add_cuts_s1(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi + hi <= ya + ma * (xi + li - xa) + cut_tol
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s1"
        )

    def _add_cuts_s2(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi + hi <= ya + ma * (xi - xa) + cut_tol
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s2"
        )

    def _add_cuts_s3(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi >= ya + ma * (xi - xa) - cut_tol
             for (xi, yi) in pos
             for ((xa, ya), ma) in zip(a, m)),
            name="s3"
        )

    def _add_cuts_s4(self, m, dims, a, pos):
        self._model.addConstrs(
            (yi >= ya + ma * (xi + li - xa) - cut_tol
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s4"
        )

    def display(self, title: str = "", plot: bool = True, show: bool = True):
        if plot:
            ax = create_new_sqauared_ax(2 * self.R)

            self.gurobi_solution.display(self.R, ax=ax, color="r")
            add_solution_rectangles(ax=ax, solution=self.ccc.feasible_solution, color="g")
            a_s = self.ccc.get_intersection_point()[self.ccc.unfeasible_set]
            if "all_tangent" in self.optimizations:
                a_s = self._prev_as
            draw_cuts(ax=ax, R=self.R, intersection_points=a_s)
            draw_intersection_points(ax=ax, intersection_points=a_s)
            if show:
                ax.set_title(title)
                plt.show()

    def _continue_optimization(self, prev_obj_val: float, prev_feasible_obj: float,
                               time_limit: float, elapsed: float) -> float:
        a = self.add_tangent_plane_cuts()
        self._prev_as = np.vstack((self._prev_as, a))

        if "objective_bound" in self.optimizations:
            self._model.Params.BestObjStop = min(self._model.Params.BestObjStop,
                                                 prev_obj_val + 1e-4)
            print(f"Objective bound: {prev_obj_val}")

        if "cutoff" in self.optimizations:
            self._model.Params.Cutoff = max(prev_feasible_obj - 1e-4, self._model.Params.Cutoff)
            print(f"Cutoff value: {prev_feasible_obj}")

        self._model.Params.TimeLimit = max(time_limit - elapsed, 0)
        return self._timed_optimize()

    def _store_feasible_solution(self, elapsed: float, feasible_solution: Solution):
        self.history.append((elapsed, feasible_solution))
        self._best_sol.update(feasible_solution)

    def _timed_optimize(self) -> float:
        """Wraps the core optimization method to add a timer"""
        start = time()
        super().optimize()
        return time() - start

    def optimize(self, max_iteration: int = 100, display_each: int = 2, time_limit: int = np.inf,
                 plot: bool = True, show: bool = True):
        if "initial_objective_bound" in self.optimizations:
            self._model.Params.BestObjStop = self._get_initial_objective_bound()

        if "initial_cutoff" in self.optimizations:
            self._model.Params.Cutoff = self._get_initial_cutoff()

        self._model.Params.TimeLimit = time_limit

        elapsed = self._timed_optimize()

        if not self.is_solved:  # the problem is infeasible in the time given
            return elapsed
        self.ccc = SolutionProcesser(self)
        self.display(title="0", plot=plot, show=show)

        # may none of the initial are acceptable in the initial position
        prev_feasible_obj = min(self._accepted_values)
        for it in range(1, max_iteration):
            prev_obj_val = self._model.objVal
            # we may have no feasible package in solution
            prev_feasible_obj = max(prev_feasible_obj, self.ccc.feasible_solution.objective)

            print(f"Iteration: {it}_________________________-")

            if self.ccc.solution_is_acceptable:
                print(f"Solution found at iteration {it}****************************************")
                self.display(title=f"Solution Found: {it}", plot=plot, show=show)
                self.optimal_solution_found = True
                break

            self._store_feasible_solution(elapsed, self.ccc.feasible_solution)

            if not self.is_solved:  # the problem is infeasible in the time given
                return elapsed

            elapsed += self._continue_optimization(prev_obj_val, prev_feasible_obj, time_limit,
                                                   elapsed)

            if not self.is_solved:  # No solution is found till the time limit
                return elapsed

            self.ccc = SolutionProcesser(self)

            if it % display_each == 0:
                self.display(title=f"{it}", plot=plot, show=show)

        return elapsed

    def _add_xy_boundaries_constr(self, x, y):
        """
        Add the boundary constraints for the x and y variables
        VI1 is to be used then the constraints are tighter because the sagittas are used.
        """
        if "sagitta" in self.optimizations:
            self._add_xy_sagitta_boundaries(x, y)
            return
        Opp._add_xy_boundaries_constr(self, x, y)

    def _add_xy_sagitta_boundaries(self, x, y):
        """
        Adding the sagitta boundaries that correspond to VI1 from the paper
        The reference equation are (27) and (28)
        """
        s_l, s_h = self._compute_sagittas()
        self._constr["27"] = self._model.addConstrs(
            (x[i] == [s_h[i], 2 * self.R - self._l[i] - s_h[i]] for i in self._items),
            name="27")
        self._constr["28"] = self._model.addConstrs(
            (y[i] == [s_l[i], 2 * self.R - self._h[i] - s_l[i]] for i in self._items),
            name="28")

    def _add_infeasible_pairs_opt(self, a):
        if "big_M" in self.optimizations:
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
    rho = 0.4

    rng = np.random.default_rng(42)
    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)

    start = datetime.now()
    print(start)
    opts = [
        # Core optimizations
        "big_M",  #

        # Valid Inequalities
        "sagitta",  # VI1
        "area",  # VI2
        "feasible_subsets",  # VI3
        "infeasible_pairs",  # VI4
        "symmetry",  # VI5

        # Cutting plane method optimizations
        "all_tangent",
        "objective_bound",
        "cutoff",
        "initial_objective_bound",
        "initial_cutoff"

        # Not implemented
        # "initial_tangent"
        # "symmetric_tangent",
    ]
    print(f"Using optimizations {opts}, {N=}, {circle_area=}, {R=}")
    cpp = Cpp(dataset=data, values="volume", radius=R, optimizations=opts,
              rotation=False,
              rotate_with_duplicates=False
              )
    cpp.optimize(100, 2, time_limit=60 * 60)
    stop = datetime.now()
    print([(t, s.objective) for (t, s) in cpp.history])
    print(start, stop, stop - start)

    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)
    cpp = Cpp(dataset=data, values="volume", radius=R, optimizations=opts,
              rotation=True,
              rotate_with_duplicates=True,
              )
    cpp.optimize(100, 2, time_limit=60 * 60)
