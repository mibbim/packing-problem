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
all_optimizations = {"big_M", "delta", "sagitta", "area", "feasible_subsets", "infeasible_pairs",
                     "symmetry", "all_tangent", "objective_bound", "cutoff",
                     "initial_objective_bound", "initial_cutoff"}


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


def measure(obj, rotation, rho, N, d, time_threshold, optimizations=all_optimizations):
    data = d[:N]
    r = compute_radius(data, rho)
    cpp = Cpp(dataset=data,
              values=obj,
              radius=r,
              rotation=rotation,
              optimizations=optimizations)
    now = datetime.now()
    os.makedirs("{now:%d-%m}/guroby".format(now=now), exist_ok=True)
    cpp._model.Params.LogFile = "{now:%d-%m}/guroby/{now:%H:%M:%S}.log".format(now=now)
    cpp._model.Params.LogToConsole = 0
    time = cpp.optimize(1000, 1000, time_limit=time_threshold, plot=False, show=False)
    finished = cpp.optimal_solution_found
    area = cpp.solution.area
    count = cpp.solution.pos.shape[0]
    obj_val = cpp.solution.obj
    logger.info(f"{obj} problem with "
                f"{rotation=}, {rho=}, {N=}, {time=}, {area=}, {count=}, {obj_val=}, {finished=}")
    logger.info(f"{[(t, s.obj) for (t, s) in cpp.history]}, {finished=}")
    return finished


def five_datasets_measure(obj, rotation, rho, N, time_threshold, datasets):
    succesfully_computed = False
    for i, d in enumerate(datasets):
        is_optimally_solved = measure(obj, rotation, rho, N, d, time_threshold)
        # At least one instance is optimally solved
        succesfully_computed = succesfully_computed or is_optimally_solved
    return succesfully_computed


def fixed_rho_all_measure(rho, datasets):
    for obj in ["volume", "count"]:
        for rotation in (False, True):
            for N in np.arange(10, 17, 2):
                continue_to_next_n = five_datasets_measure(obj, rotation, rho, N, hour, datasets)
                if not continue_to_next_n:
                    print(f"For {rho=}, {rotation=}, {obj=}, stopped at {N=}")
                    break


def fixed_N_all_measure(N, datasets):
    for obj in ["volume", "count"]:
        for rotation in (False, True):
            for rho in (0.2, 0.4, 0.6, 0.8, 1):
                continue_to_next_rho = five_datasets_measure(obj, rotation, rho, N, hour, datasets)
                if not continue_to_next_rho:
                    break


# def missing_18_measure():
#     # TODO select correct rho for the 4 combinations
#     raise NotImplementedError
#     datasets = build_datasets()
#     time_threshold = hour
#
#     cfgs = []
#     obj, rotation, rhos = "count", False, (0.2, 0.4)
#     cfgs.append({"obj": obj, "rotation": rotation, "rhos": rhos})
#
#     for obj in ["volume", "count"]:  # ["count", "volume"]:
#         for rotation in (False, True):
#             for rho in (0.2, 0.4, 0.6, 0.8, 1):  # 0.2, 0.4, 0.6, 0.8, 1
#                 continue_to_next_rho = five_datasets_measure(obj, rotation, rho, 18, time_threshold,
#                                                              datasets)
#                 if not continue_to_next_rho:
#                     break


def missing():
    datasets = build_datasets()
    fixed_rho_all_measure(1, datasets)
    # fixed_N_all_measure(8, datasets)


def main():
    datasets = build_datasets()
    time_threshold = hour
    already_solved = True
    for obj in ["volume"]:  # ["count", "volume"]:
        for rotation in (True,):  # False,
            max_n = 17
            if rotation:
                max_n = 17

            for rho in (0.2, 0.4, 0.6, 0.8, 1):  # 0.2, 0.4, 0.6, 0.8, 1
                # for rho in (0.2, 0.4, 0.6, 0.8, 1):  # 0.2, 0.4, 0.6, 0.8, 1
                continue_to_next_rho = False
                for N in np.arange(10, max_n, 2):  # with 18 sometimes goes out of memory

                    if rho == 0.8 and N == 12:  # start from this cfg
                        already_solved = False
                        continue_to_next_rho = False
                    if already_solved:
                        continue_to_next_rho = True
                        continue

                    continue_to_next_n = five_datasets_measure(obj, rotation, rho, N,
                                                               time_threshold, datasets)

                    if not continue_to_next_n:
                        print(f"For {rho=}, stopped at {N=}")
                        max_n = N
                        break

                # Only for skipping already solved

                # except AttributeError:
                #     logger.info(
                #         f"Attribute Error Occured at {obj=} {rotation=}, {rho=}, {N=}   {i=}")

        pass


logger = logging.getLogger()
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s %(lineno)d][%(levelname)s] %(message)s",
                      datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(stdout_handler)
if __name__ == "__main__":
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
    # main()
    missing()
