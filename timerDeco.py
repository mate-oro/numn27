import time

def timeit(func):
    """Decorator to measure the execution time of a function.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: The decorated function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.3f} seconds to execute.")
        return result

    return wrapper