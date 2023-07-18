import time
from smokingml.utils import Colors
from typing import Callable, TypeVar, ParamSpec

T = TypeVar('T')
P = ParamSpec('P')

def print_on_start_and_end(func: Callable[P, T]) -> Callable[P, T]:
    
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print(f'{Colors.OKGREEN}Starting {func.__name__}{Colors.ENDC}')
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print(f'{Colors.WARNING}Finished {func.__name__}. Elapsed time: {end-start:.3f}{Colors.ENDC}')
        return ret
    
    return wrapper