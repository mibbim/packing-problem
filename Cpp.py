from typing import Tuple, List, Literal

import matplotlib.pyplot as plt
import numpy as np

from Opp import Opp
from RPP import Rpp
from circular_container_cuts import CircularContainerCuts


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
        self.ccc = None
        self._prev_as = np.empty((0, 2))

    @staticmethod
    def _get_feasible_items(r, dataset):
        """Needs R"""
        return dataset[np.linalg.norm(dataset, axis=1) <= 2 * r]

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

        if "all_tangent" in self.optimizizations:
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
        # for a
        if show:
            plt.title(title)
            plt.show()

    def optimize(self, max_iteration: int = 10, display_each: int = 2):
        Rpp.optimize(self)
        self.display(title="0")
        prev_obj_val = self._model.objVal
        self.ccc = CircularContainerCuts(self)
        for it in range(1, max_iteration):
            print(f"Iteration: {it}_________________________-")
            if self.ccc.acceptable:
                print(f"Solution found at iteration {it}******************************************")
                self.display(title=f"Solution Found: {it}")
                break
            a = self.add_tangent_plane_cuts()
            # if "all_tangent" in self.optimizizations:
            self._prev_as = np.vstack((self._prev_as, a))
            if "cutoff" in self.optimizizations:
                self._model.Params.BestObjStop = prev_obj_val
                # self._model.Params.BestBdStop = prev_obj_val
            super().optimize()
            if it % display_each == 0:
                self.display(title=f"{it}")
            self.ccc = CircularContainerCuts(self)
        # self.display()
        print()

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

    def _total_area(self):
        return self.R * self.R * np.pi


if __name__ == "__main__":
    from datetime import datetime

    N = 10
    rho = 0.33

    np.random.seed(42)
    rng = np.random.default_rng()
    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)
    # R = 1.5
    # data = [(0.6, 1.2) for _ in range(10)]
    # data = [(0.6, 1.2) for _ in range(30)]
    start = datetime.now()
    print(start)
    opts = [
        "big_M",
        "delta",
        "sagitta",
        "area",
        "all_tangent",
        "cutoff", "feasible_subsets"
    ]
    cpp = Cpp(dataset=data, values="volume", radius=R, optimizations=opts)
    cpp.optimize(40, 2)
    stop = datetime.now()
    print(start, stop, stop - start)
