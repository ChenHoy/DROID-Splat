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
    if isinstance(obj, tuple):
        return copy.deepcopy(obj)

    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(getattr(clone_obj.__class__, attr), property):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj


def get_all_queue(queue: mp.Queue):
    """Get all elements from the queue until it is empty."""
    result = []
    while not queue.empty():
        latest = queue.get_nowait()
        if latest is not None:
            result.append(clone_obj(latest))
        del latest
    return result
