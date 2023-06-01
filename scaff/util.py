from itertools import islice
from typing import Iterable 

def batched(iterable: Iterable, n:int) -> iter:
    """
    Batch data into tuples of length n. The last batch may be shorter.
    Direct copy from the python itertools documentation

    Args:
        iterable (Iterable): Input iterable.
        n (int): The size of the batches.

    Returns:
        Iterator[Tuple]: An iterator with the input data divided into batches.

    Example:
        batched('ABCDEFG', 3) --> ABC DEF G
    """
    
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch