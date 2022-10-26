class Solution:

    def __init__(self, objective, positions, dimensions):
        self._obj = objective
        self._positions = positions
        self._dims = dimensions

    @property
    def obj(self):
        return self._obj

    @property
    def pos(self):
        return self._positions

    @property
    def dims(self):
        return self._dims

    def as_dict(self):
        return {
            "obj": self.obj,
            "pos": self.pos,
            "dims": self.dims,
        }


class BestSolution(Solution):
    def update(self, new_solution: Solution):
        if new_solution.obj > self.obj:
            self._obj = new_solution.obj
            self._positions = new_solution.pos
            self._dims = new_solution.dims
