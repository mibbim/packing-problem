import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.Cpp import Cpp

NPA = np.ndarray

script_dir = Path(__file__).resolve().parent

hour = 3600
all_optimizations = [
    "big_M",  #
    "delta",  #
    "sagitta",  #
    "area",  #
    "feasible_subsets",  #
    "infeasible_pairs",  #
    "symmetry",  #
    "all_tangent",
    "objective_bound",
    "cutoff",
    "initial_objective_bound",
    "initial_cutoff",
]


def build_datasets(instances: int = 5, n: int = 30, seed: int = 42) -> NPA:
    rng = np.random.default_rng(seed)
    dims = 2
    datasets = 4 * rng.random((instances, n, dims)) + 1
    return datasets


def compute_radius(dataset: NPA, rho: float):
    """Operates on a single instance"""
    packs_area = np.prod(dataset, axis=1).sum()
    circle_area = rho * packs_area
    return np.sqrt(circle_area / np.pi)


def measure(obj, rotation, rho, N, d, time_threshold, ):
    data = d[:N]
    r = compute_radius(data, rho)
    cpp = Cpp(dataset=data,
              values=obj,
              radius=r,
              rotation=rotation,
              optimizations=all_optimizations)
    now = datetime.now()
    os.makedirs("{now:%d-%m}/guroby".format(now=now), exist_ok=True)
    cpp._model.Params.LogFile = "{now:%d-%m}/guroby/{now:%H:%M:%S}.log".format(now=now)
    cpp._model.Params.LogToConsole = 0
    time = cpp.optimize(1000, 1000, time_limit=time_threshold, plot=False, show=False)
    area = cpp.area
    count = cpp.count
    obj_val = cpp.obj_val
    logger.info(f"{obj} problem with "
                f"{rotation=}, {rho=}, {N=}, {time=}, {area=}, {count=}, {obj_val=}")


def main():
    datasets = build_datasets()
    time_threshold = hour
    for obj in ["count", "volume"]:
        for rotation in (True, False):
            for rho in (0.2, 0.4, 0.6, 0.8, 1):
                for N in np.arange(10, 18, 1):
                    for i, d in enumerate(datasets):
                        try:
                            measure(obj, rotation, rho, N, d, time_threshold)
                        except AttributeError:
                            logger.info(
                                f"Attribute Error Occured at {obj=} {rotation=}, {rho=}, {N=}   {i=}")


        pass


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s %(lineno)d][%(levelname)s] %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(stdout_handler)

    logfile_handler = logging.FileHandler("train.log")
    logfile_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(module)s %(lineno)d][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(logfile_handler)

    # print(isinstance(AttributeError, RuntimeError))
    # for d in build_datasets():
    #     measure(obj="count", rotation=True, rho=0.2, N=15, d=d, time_threshold=hour)
    main()
