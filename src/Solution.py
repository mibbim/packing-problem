import matplotlib.pyplot as plt
import numpy as np


class Solution:

    def __init__(self, positions=np.empty((0, 2)),
                 dimensions=np.empty((0, 2), ),
                 values=np.empty((0, 1)),
                 rotated=None,
                 ):
        self._positions = positions
        self._dims = dimensions
        self._values = values
        self._rotated = rotated
        if rotated is None:
            self._rotated = np.zeros((positions.shape[0],), dtype=bool)

    @property
    def obj(self):
        return self._values.sum()

    @property
    def area(self):
        return self._dims.prod(axis=1).sum()

    @property
    def pos(self):
        return self._positions

    @property
    def dims(self):
        return self._dims

    @property
    def rotated(self):
        return self._rotated

    @property
    def values(self):
        return self._values

    def as_dict(self):
        return {
            "obj": self.obj,
            "pos": self.pos,
            "dims": self.dims,
            "values": self._values,
            "rotated": self.rotated
        }

    def display(self, R: float, ax=None, color=None):
        if ax is None:
            fig, ax = plt.subplots()
        for i, (pos, dim, r) in enumerate(zip(self.pos, self.dims, self.rotated)):
            if r:
                dim = dim[::-1]
            draw_rectangle_with_number_in_center(ax, pos[0], pos[1], dim[0], dim[1], i, color=color)

        draw_circle(ax, R, R, R)


def draw_circle(ax, x, y, radius, color="c"):
    circle = plt.Circle((x, y), radius, color=color, fill=True, alpha=0.2)
    ax.add_artist(circle)


def draw_rectangle(ax, x, y, width, height, color="g", linewidth: float = 1):
    rectangle = plt.Rectangle((x, y), width, height,
                              linewidth=linewidth, edgecolor=color,
                              facecolor='none')
    ax.add_artist(rectangle)


def draw_rectangle_with_number_in_center(ax, x, y, width, height, number, color="g"):
    draw_rectangle(ax, x, y, width, height, color=color, linewidth=0.5)
    ax.annotate(str(number), xy=(x + width / 2, y + height / 2))


def add_solution_rectangles(ax, solution: Solution, color="g"):
    for (pos, dims, r) in zip(solution.pos, solution.dims, solution.rotated):
        if r:
            dims = dims[::-1]
        draw_rectangle(ax, pos[0], pos[1], dims[0], dims[1], color=color)


class BestSolution(Solution):
    def update(self, new_solution: Solution):
        if new_solution.obj > self.obj:
            self._values = new_solution.values
            self._positions = new_solution.pos
            self._dims = new_solution.dims
            self._rotated = new_solution.rotated
