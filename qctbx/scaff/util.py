from itertools import islice
from functools import reduce
from copy import deepcopy
from typing import Iterable, Dict, Any

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

def dict_merge(*args: Dict[str, Any], case_sensitive: bool=True) -> Dict[str, Any]:
    """
    Merge multiple input dictionaries into a single dictionary.
    
    This function handles nested dictionaries and lists within those 
    dictionaries. It merges keys from each dictionary into the first one. 
    If a key exists in both dictionaries and both values are dictionaries, 
    the function recursively merges the nested dictionaries. If both 
    values are lists, it combines them, removing duplicates. If the values 
    are of the same type, it overwrites the value in the first dictionary 
    with the value from the second. If the key exists in the second 
    dictionary but not the first, it adds the key-value pair to the first 
    dictionary.

    Note: The function raises an exception if it encounters mismatched 
    value types for the same key in different dictionaries.

    Args:
        *args: A variable number of dictionaries to be merged.
        case_sensitive (bool): if True case sensitivity will be obeyed in the 
            merging of dictionaries and lists. If False, the merging will be 
            done with the assumption that different case strings are equal. 
            Resulting strings will come from the latest dict in args.

    Returns:
        A single dictionary that is the result of merging all input 
        dictionaries.

    Raises:
        Exception: If a key in the dictionaries has mismatched value types.

    Adapted from: 
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    """

    def inner_merge(a, b, path=None):
        "merges b into a"

        def lower_if_str(value):
            if type(value) == str:
                return value.lower()
            else:
                return value

        if path is None: path = []
        if not case_sensitive:
            a_compare = list(lower_if_str(key) for key in a.keys())
        else:
            a_compare = a.keys()
        orig_a_keys = list(a.keys())
        for b_key in b:
            if case_sensitive:
                b_compare = b_key
            else:
                b_compare = lower_if_str(b_key)
            if b_compare in a_compare:
                if case_sensitive:
                    a_key = b_key
                else:
                    a_key = orig_a_keys[a_compare.index(lower_if_str(b_key))]
                if isinstance(a[a_key], dict) and isinstance(b[b_key], dict):
                    inner_merge(a[a_key], b[b_key], path + [str(b_key)])
                elif isinstance(a[a_key], list) and isinstance(b[b_key], list):
                    if case_sensitive:
                        a[b_key] = list(set(a[a_key] + b[b_key]))
                    else:
                        b_lower = list(map(lower_if_str, b[b_key]))
                        a_keep = [val for val in a[a_key] if lower_if_str(val) not in b_lower]
                        del(a[a_key])
                        a[b_key] = b[b_key] + a_keep
                elif type(a[a_key]) == type(b[b_key]):
                    if not case_sensitive:
                        del(a[a_key])
                    a[b_key] = b[b_key]
                else:
                    raise Exception(f"Types of entries do not match at {'.'.join(path + [str(b_key)])}, type1 {str(type(b[b_key]))}, type2 {str(type(a[a_key]))}")
            else:
                a[b_key] = b[b_key]
        return a
    
    return reduce(inner_merge, iter(deepcopy(arg) if i == 0 else arg for i, arg in enumerate(args)))