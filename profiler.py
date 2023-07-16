# -*- coding: utf-8 -*-
"""
Centralise the profiling operations of the project
"""
import pstats
import pathlib


def profile_script(top_logs: int = 100):
    """
    Simple profiling of the environment
    Assumes the following cmd has run: python -m cProfile -o [local_path_to]/restats [script]
    """

    profiler = pstats.Stats(str(pathlib.Path(__file__).parent.joinpath("restats")))
    profiler.strip_dirs().sort_stats("tottime").print_stats(top_logs)


if __name__ == "__main__":
    profile_script()
