# -*- coding: utf-8 -*-
"""
from typing import List, Tuple
def buffer():
    import time
    time.sleep(1)
    return None
def this_is_some_processing(x: int) -> List:
    assert 1 < x < 10
    for i in range(1, x):
        buffer()
    if x % 2 == 0:
        print(f"{x} this is an even number. converting to odd now !")
        x = str(x)
        return (x)
    else:
        print(f"{x} this is an odd number. converting to odd now !")
        x = str(x)
        return (x)
if __name__ == '__main__':
    import numpy as np
    TEST_VALUE = np.random.choice([3, 4, 5])
    f_v = this_is_some_processing(TEST_VALUE)
    print(f_v)

Objective:
1. Formatting of the code with Black
2. Linting of the code with Pylint
3. Typing of the code with MyPy
4. Fixing of the code with Profiler
"""
# type: ignore
# pylint: skip-file
