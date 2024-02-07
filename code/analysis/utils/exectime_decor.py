import time
from functools import wraps

def timeit(func):

    """
    Measure and print execution time of given function

    Inputs:
    - func : function to call and measure

    Outputs:
    - result : function results
    - prints execution time
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper