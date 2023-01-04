import numpy as np

from src.Cpp import Cpp, cut_tol


class Cpp_rot(Cpp):
    """ Orthogonal packing problem implementation that allows rotation without duplication."""

    def _optimize_M(self, M):
        if not self.rotation:
            return super()._optimize_M(M)
        s_l, s_h = self._compute_sagittas()
        min_s = np.minimum(s_l, s_h)
        s_min_sum = np.add.outer(min_s, min_s)
        M[:] -= np.expand_dims(s_min_sum, axis=-1)
        return M

    def _add_xy_sagitta_boundaries(self, x, y):
        if not self.rotation:
            return super()._add_xy_sagitta_boundaries(x, y)
        r = self._r
        s_l, s_h = self._compute_sagittas()
        l, h = self._l, self._h

        self._constr["29_1"] = self._model.addConstrs(
            (x[i] <= 2 * self.R - ((1 - r[i]) * (l[i] + s_h[i]) +
                                   r[i] * (h[i] + s_l[i])) for i in self._items),
            name="29_1")
        self._constr["29_2"] = self._model.addConstrs(
            (x[i] >= (1 - r[i]) * s_h[i] + r[i] * s_l[i] for i in self._items),
            name="29_2")
        self._constr["30_1"] = self._model.addConstrs(
            (y[i] <= 2 * self.R - ((1 - r[i]) * (h[i] + s_l[i]) +
                                   r[i] * (l[i] + s_h[i])) for i in self._items),
            name="30_1")
        self._constr["30_2"] = self._model.addConstrs(
            (y[i] >= (1 - r[i]) * s_l[i] + r[i] * s_h[i] for i in self._items),
            name="30_2")

    def _add_infeasible_pairs_opt(self, a):
        if not self.rotation:
            return super()._add_infeasible_pairs_opt(a)
        s_l, s_h = self._compute_sagittas()
        min_s = np.minimum(s_l, s_h)
        min_dim = self.dims.min(axis=1)
        min_s_sum = np.add.outer(min_s, min_s)
        min_dim_sum = np.add.outer(min_dim, min_dim)
        infeasible = np.triu(min_dim_sum >= 2 * self.R - min_s_sum, k=1)
        self._model.addConstrs(
            (a[i] + a[j] <= 1 for i, j in self._items_combinations if infeasible[i, j]),
            name="infeasible_pairs"
        )

    def _add_cuts_s2(self, m, dims, a, pos):
        if not self.rotation:
            return super()._add_cuts_s2(m, dims, a, pos)
        r = self._r
        # hi = hi * (1 - ri) + li * ri
        # li = li * (1 - ri) + hi * ri
        self._constr["s2"] = self._model.addConstrs(
            (yi + hi * (1 - ri) + li * ri <= ya + ma * (xi - xa) + cut_tol
             for (xi, yi), (li, hi), ri in zip(pos, dims, r)
             for ((xa, ya), ma) in zip(a, m)),
            name="s2"
        )

    def _add_cuts_s3(self, m, dims, a, pos):
        if not self.rotation:
            return super()._add_cuts_s3(m, dims, a, pos)
        self._constr["s3"] = self._model.addConstrs(
            (yi >= ya + ma * (xi - xa) - cut_tol
             for (xi, yi) in pos
             for ((xa, ya), ma) in zip(a, m)),
            name="s3"
        )

    def _add_cuts_s4(self, m, dims, a, pos):
        if not self.rotation:
            return super()._add_cuts_s4(m, dims, a, pos)
        r = self._r
        self._constr["s4"] = self._model.addConstrs(
            (yi >= ya + ma * (xi + li * (1 - ri) + hi * ri - xa) - cut_tol
             for (xi, yi), (li, hi), ri in zip(pos, dims, r)
             for ((xa, ya), ma) in zip(a, m)),
            name="s4"
        )


if __name__ == "__main__":
    N = 10  # 20
    rho = 0.4

    rng = np.random.default_rng(42)
    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)

    opts = [
        # Core optimizations
        "big_M",  #
        #
        # # Valid Inequalities
        "sagitta",  # VI1
        "area",  # VI2
        "feasible_subsets",  # VI3
        "infeasible_pairs",  # VI4
        "symmetry",  # VI5

        # Cutting plane method optimizations
        # "all_tangent",
        "objective_bound",
        "cutoff",
        "initial_objective_bound",
        "initial_cutoff"

        # Not implemented
        # "initial_tangent"
        # "symmetric_tangent",
    ]
    cpp = Cpp_rot(dataset=data, values="volume", radius=R, optimizations=opts,
                  rotation=False,
                  rotate_with_duplicates=False)

    cpp.optimize()
    print("rotation")
    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)
    data = data[:5]
    # print()
    cpp_rot = Cpp_rot(dataset=data, values="volume", radius=R, optimizations=opts,
                      rotation=True,
                      rotate_with_duplicates=False
                      )
    cpp_rot.optimize(display_each=100)
    print("rotation with duplicates")
    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)
    cpp_dup = Cpp_rot(dataset=data, values="volume", radius=R, optimizations=opts + ["all_tangent"],
                      rotation=True,
                      rotate_with_duplicates=True
                      )
    cpp_dup.optimize(display_each=100)
