from typing import List
from .image_utils import *
from .multiprocessing_utils import *
from .system_utils import *
from .loop_utils import *


def reduce_to_single_list(list_of_lists: List[List]) -> List:
    reduced_list = []
    for el in list_of_lists:
        if isinstance(el, list):
            reduced_list.extend(reduce_to_single_list(el))
        else:
            reduced_list.append(el)
    return reduced_list
