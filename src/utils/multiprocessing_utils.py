import copy

import torch
import torch.multiprocessing as mp


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise mp.queues.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def clone_obj(obj):
    """Clone all torch.Tensor objects in an object."""

    cloned_obj = copy.deepcopy(obj)

    # Take care of List[torch.Tensor] or Tuple[torch.Tensor]
    if isinstance(cloned_obj, list) or isinstance(cloned_obj, tuple):
        for i, el in enumerate(cloned_obj):
            if isinstance(el, torch.Tensor):
                cloned_obj[i] = el.detach().clone()
        return cloned_obj

    # Go detach and clone Dict[str, torch.Tensor]
    if not hasattr(cloned_obj, "__dict__"):
        return cloned_obj
    for attr in cloned_obj.__dict__.keys():
        # check if its a property
        if hasattr(cloned_obj.__class__, attr) and isinstance(getattr(cloned_obj.__class__, attr), property):
            continue
        if isinstance(getattr(cloned_obj, attr), torch.Tensor):
            setattr(cloned_obj, attr, getattr(cloned_obj, attr).detach().clone())
    return cloned_obj


def get_all_queue(queue: mp.Queue):
    """Get all elements from the queue until it is empty."""
    result = []
    while not queue.empty():
        latest = queue.get_nowait()
        if latest is not None:
            result.append(clone_obj(latest))
        del latest
    return result
