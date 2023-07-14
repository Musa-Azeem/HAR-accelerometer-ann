import time
from smokingml.utils import Colors
from typing import Callable

def print_on_start_and_end(func: Callable) -> Callable:
    def wrapper(*args: any, **kwargs: any) -> None:
        print(f'{Colors.OKGREEN}Starting {func.__name__}{Colors.ENDC}')
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f'{Colors.WARNING}Finished {func.__name__}. Elapsed time: {end-start:.3f}{Colors.ENDC}')
        return ret
    return wrapper