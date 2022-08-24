from typing import Tuple, List, Literal

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
        super().__init__(dataset, values, radius, rotation, optimizations, name)
        self.ccc = None

    def add_tangent_plane_cuts(self):
        ccc = self.ccc
        a = ccc.get_intersection_point()
        a_minus_r = a - self.R
        m = - a_minus_r[:, 0] / a_minus_r[:, 1]
        accepted = self.accepted
        self.reset()
        add_cuts_methods = [self._add_cuts_s1,
                            self._add_cuts_s2,
                            self._add_cuts_s3,
                            self._add_cuts_s4]

        for i, add_cut_method in enumerate(add_cuts_methods):
            add_cut_method(m[ccc.s[i]], ccc.accepted_dims[ccc.s[i]],
                           a[ccc.s[i]], self.pos[accepted][ccc.s[i]])

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

    def optimize(self, max_iteration: int = 10, display_each: int = 2):
        Rpp.optimize(self)
        self.ccc = CircularContainerCuts(self)
        for it in range(max_iteration):
            if self.ccc.acceptable:
                self.display()
                print(f"Solution found at iteration {it}******************************************")
                break
            self.add_tangent_plane_cuts()
            Rpp.optimize(self)
            if it % display_each == 0:
                self.display()
            self.ccc.compute_from_model(self)


if __name__ == "__main__":
    R = 1.5
    data = [(0.6, 1.2) for _ in range(10)]
    cpp = Cpp(dataset=data, values="volume", radius=R, )
    cpp.optimize()
