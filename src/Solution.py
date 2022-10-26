class Solution:

    def __init__(self, positions, dimensions, values):
        self._values = values
        self._positions = positions
        self._dims = dimensions

    @property
    def obj(self):
        return self._values.sum()

    @property
    def pos(self):
        return self._positions

    @property
    def dims(self):
        return self._dims

    @property
    def values(self):
        return self._values

    def as_dict(self):
        return {
            "obj": self.obj,
            "pos": self.pos,
            "dims": self.dims,
            "values": self._values
        }


class BestSolution(Solution):
    def update(self, new_solution: Solution):
        if new_solution.obj > self.obj:
            self._values = new_solution.values
            self._positions = new_solution.pos
            self._dims = new_solution.dims
