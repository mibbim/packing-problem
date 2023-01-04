from src.Cpp import Cpp, compute_tangent_angular_coefficient


class Cpp_rot(Cpp):
    """ Orthogonal packing problem implementation that allows rotation."""

    pass

    def _optimize_M(self, M):
        raise NotImplementedError

    def _add_xy_sagitta_boundaries(self, x, y):
        raise NotImplementedError

    def _add_infeasible_pairs_opt(self, a):
        raise NotImplementedError

    def add_tangent_plane_cuts(self):
        if not self.rotation:  # and not self._duplicate: it is equivalent
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
