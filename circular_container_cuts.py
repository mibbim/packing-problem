from abc import ABC
import numpy as np
import RPP

NPA = np.ndarray


class CircularContainerCuts(ABC):
    def __init__(self, model: RPP):
        assert model.is_solved
        radius, accepted_pos, accepted_dims = model.R, model.accepted_pos, model.accepted_dims
        self.model = model
        self._R: float = model.R
        self._accepted_pos: NPA = model.accepted_pos
        self._accepted_dims: NPA = model.accepted_dims
        # self._bar_set, self._point_to_check, self._s = self.compute_variables()
        self._bar_set: NPA = self._get_bar_set()
        self._point_to_check: NPA = self._get_point_to_check()
        self._s: NPA = self._get_s()
        self.acceptable = not bool(self._s.sum())

    def compute_from_model(self, model=None):
        if model is None:
            model = self.model
        return self.__class__(model)

    def add_cuts(self):
        # r = self.model.R
        a = self.get_intersection_point()
        a_minus_r = a - self.model.R
        # l, h = self._accepted_dims[:, 0], self._accepted_dims[:, 1]
        m = - a_minus_r[:, 0] / a_minus_r[:, 1]
        accepted = [i in self.model.accepted for i in range(len(self.model.pos))]
        self.model.reset()
        add_cuts_methods = [self._add_cuts_s1,
                            self._add_cuts_s2,
                            self._add_cuts_s3,
                            self._add_cuts_s4]
        for i, add_cut_method in enumerate(add_cuts_methods):
            add_cut_method(self.model, m[self._s[i]], self._accepted_dims[self._s[i]],
                           a[self._s[i]], self.model.pos[accepted][self._s[i]])

    @staticmethod
    def _add_cuts_s1(model, m, dims, a, pos):
        model.addConstrs(
            (yi + hi <= ya + ma * (xi + li - xa)
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s1"
        )

    @staticmethod
    def _add_cuts_s2(model, m, dims, a, pos):
        model.addConstrs(
            (yi + hi <= ya + ma * (xi - xa)
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s2"
        )

    @staticmethod
    def _add_cuts_s3(model, m, dims, a, pos):
        model.addConstrs(
            (yi >= ya + ma * (xi - xa)
             for (xi, yi) in pos
             for ((xa, ya), ma) in zip(a, m)),
            name="s3"
        )

    @staticmethod
    def _add_cuts_s4(model, m, dims, a, pos):
        model.addConstrs(
            (yi >= ya + ma * (xi + li - xa)
             for (xi, yi), (li, hi) in zip(pos, dims)
             for ((xa, ya), ma) in zip(a, m)),
            name="s4"
        )

    def _get_s(self):
        return self._is_oob(self._point_to_check) & self._bar_set
        # return np.array([s & self._is_oob(self._point_to_check) for s in self._bar_set])

    def _get_bar_set(self) -> NPA:
        """
        needs self._accepted_pos, self._accepted_dims, self._R
        :return:
        """
        bar = (self._accepted_pos + self._accepted_dims * 0.5)
        left_idx = bar[:, 0] <= self._R
        right_idx = np.logical_not(left_idx)
        bar_set = np.array(
            (right_idx & (bar[:, 1] > self._R),
             left_idx & (bar[:, 1] > self._R),
             left_idx & (bar[:, 1] <= self._R),
             right_idx & (bar[:, 1] <= self._R),
             )
        )
        return bar_set

    def _is_oob(self, points_to_check):
        return np.linalg.norm(points_to_check - self._R, axis=1) > self._R

    def get_intersection_point(self):
        c_meno_o = self._R - self._point_to_check
        step_magnitude = (1 - self._R / np.linalg.norm(c_meno_o, axis=1))[:, np.newaxis]
        return self._point_to_check + c_meno_o * step_magnitude

    def _get_point_to_check(self) -> NPA:
        """Needs self._accepted_pos, self._accepted_dims, self._bar_set"""
        point_to_check = self._accepted_pos.copy()
        point_to_check[self._bar_set[0]] += self._accepted_dims[self._bar_set[0]]
        point_to_check[:, 1][self._bar_set[1]] += self._accepted_dims[:, 1][self._bar_set[1]]
        point_to_check[:, 0][self._bar_set[3]] += self._accepted_dims[:, 0][self._bar_set[3]]
        return point_to_check


if __name__ == "__main__":
    from RPP import Rpp

    plot = False
    R = 1.5
    data = [(0.6, 1.2) for _ in range(10)]
    rpp = Rpp(dataset=data, values="volume", radius=R)
    rpp.optimize()
    rpp.display()
    pos, dims = rpp.accepted_pos, rpp.accepted_dims
    # pos[:, 1] += 1
    ccc = CircularContainerCuts(rpp)
    A = ccc.get_intersection_point()
    P = ccc._point_to_check

    ccc.add_cuts()
    rpp.optimize()
    rpp.display()

    for it in range(50):
        ccc = ccc.compute_from_model()
        if ccc.acceptable:
            print(
                f"Solution found at iteration {it}*****************************************************************************************************")
            rpp.display()
            break
        ccc.add_cuts()
        rpp.optimize()
        if it % 5:
            rpp.display()
    rpp2 = Rpp(dataset=data, values="volume", radius=R)

    print()
