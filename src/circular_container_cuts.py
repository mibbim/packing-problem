from abc import ABC
import numpy as np
from src.RPP import Rpp

NPA = np.ndarray


class CircularContainerCuts(ABC):
    def __init__(self, model: Rpp):
        assert model.is_solved
        self.model = model
        self._R: float = model.R
        self.accepted_pos: NPA = model.accepted_pos
        self.accepted_dims: NPA = model.accepted_dims
        self._bar_set: NPA = self._get_bar_set()
        self._point_to_check: NPA = self._get_point_to_check()
        self._s: NPA = self._get_s()
        self.acceptable = not bool(self._s.sum())
        # self.values = model.values
        self.accepted_values = model.accepted_values

    @property
    def s(self):
        return self._s

    @property
    def unfeasible_set(self):
        return self.s.sum(axis=0).astype(bool)

    @property
    def feasible_set(self):
        return np.logical_not(self.unfeasible_set)

    @property
    def feasible_area(self):
        return self.accepted_dims[self.feasible_set].prod(axis=1).sum()

    @property
    def feasible_obj(self):
        return self.accepted_values[self.feasible_set].sum()

    def _get_s(self):
        return self._is_oob(self._point_to_check) & self._bar_set

    def _get_bar_set(self) -> NPA:
        """
        needs self._accepted_pos, self._accepted_dims, self._R
        :return:
        """
        bar = (self.accepted_pos + self.accepted_dims * 0.5)
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

    def _is_oob(self, points_to_check, tol: float = 1e-3):
        return np.linalg.norm(points_to_check - self._R, axis=1) - self._R > tol

    def get_intersection_point(self):
        c_meno_o = self._R - self._point_to_check
        step_magnitude = (1 - self._R / np.linalg.norm(c_meno_o, axis=1))[:, np.newaxis]
        return self._point_to_check + c_meno_o * step_magnitude

    def _get_point_to_check(self) -> NPA:
        """Needs self._accepted_pos, self._accepted_dims, self._bar_set"""
        point_to_check = self.accepted_pos.copy()
        point_to_check[self._bar_set[0]] += self.accepted_dims[self._bar_set[0]]
        point_to_check[:, 1][self._bar_set[1]] += self.accepted_dims[:, 1][self._bar_set[1]]
        point_to_check[:, 0][self._bar_set[3]] += self.accepted_dims[:, 0][self._bar_set[3]]
        return point_to_check


if __name__ == "__main__":
    from RPP import Rpp

    plot = False
    R = 1.5
    data = np.array([(0.6, 1.2) for _ in range(10)])
    rpp = Rpp(dataset=data, values="volume", radius=R)
    rpp.optimize()
    rpp.display()
    pos, dims = rpp.accepted_pos, rpp.accepted_dims
    # pos[:, 1] += 1
    ccc = CircularContainerCuts(rpp)
    A = ccc.get_intersection_point()
    P = ccc._point_to_check

    # ccc.add_cuts()
    rpp.optimize()
    rpp.display()

    print()
