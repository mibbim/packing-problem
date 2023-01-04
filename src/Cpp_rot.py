import numpy as np

from src.Cpp import Cpp, compute_tangent_angular_coefficient


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

    def add_tangent_plane_cuts(self):
        if not self.rotation:  # and not self._duplicate: it is equivalent
            return super().add_tangent_plane_cuts()
        if self._duplicate:
            return super().add_tangent_plane_cuts()
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
                add_cut_method(m[ccc.s[i]], self.dims[:, ::-1],
                               a[ccc.s[i]], self.pos)
            cuts_added = ccc.s.sum() * self.dims.shape[0] * 2
        else:
            for i, add_cut_method in enumerate(add_cuts_methods):
                add_cut_method(m[ccc.s[i]], ccc.accepted_dims[ccc.s[i]],
                               a[ccc.s[i]], self.pos[accepted][ccc.s[i]])
                add_cut_method(m[ccc.s[i]], ccc.accepted_dims[ccc.s[i]][:, ::-1],
                               a[ccc.s[i]], self.pos[accepted][ccc.s[i]])
            cuts_added = ccc.s.sum() * 2
        print(f"Adding {cuts_added} cuts")
        return a[ccc.s.sum(axis=0).astype(bool)]


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
    cpp = Cpp_rot(dataset=data, values="volume", radius=R, optimizations=opts,
                  rotation=False,
                  rotate_with_duplicates=False)

    cpp.optimize()

    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)
    cpp_rot = Cpp_rot(dataset=data, values="volume", radius=R, optimizations=opts,
                      rotation=True,
                      rotate_with_duplicates=False
                      )
    cpp_rot.optimize()

    data = 4 * rng.random((N, 2)) + 1
    packs_area = np.sum(data[:, 0] * data[:, 1])
    circle_area = rho * packs_area
    R = np.sqrt(circle_area / np.pi)
    cpp_dup = Cpp_rot(dataset=data, values="volume", radius=R, optimizations=opts,
                      rotation=True,
                      rotate_with_duplicates=True
                      )
    cpp_dup.optimize()
