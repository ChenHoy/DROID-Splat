#! /usr/bin/env python3

"""
Problem: 
- We use a video datastructure which has tensors attached to it. Different Processes read and write to this structure and perform different computations. 
- We use a shared memory to store the tensors on the GPU
- Using custom CUDA kernels with .synch_threads_(); can guarantee that the threads are synchronized, i.e. we always have the most recent written data when reading from a separate thread.
- This somehow does not work in pure Python! We only can achieve this with additional Queues, sending data back and forth between the Processes

Question: 
+ How does shared memory actually work, i.e. can we really read/write from different processes?
+ Can we somehow achieve a similar functionality like the CUDA kernel without needing to use a Queue to communicate between each Process? (we have up to 4 Processes running in parallel)
"""

from termcolor import colored
from time import sleep

import torch
import torch.multiprocessing as mp


class DataStructure:
    """Basic datastructure like DepthVideo."""

    def __init__(self, buffer_size: int = 32, device="cuda:0"):
        self.a = 3 * torch.ones(buffer_size, device=device, dtype=torch.float32).share_memory_()
        self.b = 2 * torch.ones(buffer_size, device=device, dtype=torch.float32).share_memory_()
        self.device = device

        self.counter = mp.Value("i", 0)

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        self.a[index] = item[0]
        if item[1] is not None:
            self.b[index] = item[1]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index > 0:
                index = self.counter.value + index
            item = (self.a[index], self.b[index])
        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


def process1(rank: int, data: DataStructure, val: int = 7):
    i = 0
    while True:
        el = torch.tensor(val, device=data.device)
        with data.get_lock():
            data[i] = (el, None)  # Set data.a

            # Use a newly initialized tensor change data.
            c = torch.ones_like(data.b, device=data.device)
            data.b += data.b + c

        info(f"[Process 1] Set data.a[{i}] = {val}", "red")
        info(f"[Process 1] Incremented data.b += {c[i].int()}", "red")
        sleep(5)
        i += 1


def process2(rank: int, data: DataStructure, val: int = 11):
    while True:
        with data.get_lock():
            # Just print the data to see if we can see the changes from another Process
            info(f"[Process 2] Data: a: {data.a} \n", "blue")
            info(f"[Process 2] Data: b: {data.b}", "blue")
        sleep(4)
        pass


def info(msg: str, color: str = "green"):
    """Pretty print"""
    print(colored(msg, color))


def main():
    torch.multiprocessing.set_start_method("spawn")

    data = DataStructure()
    info("Create datastructure:  a=3, b=2")

    processes = [
        mp.Process(target=process1, args=(0, data, 7), name="1"),
        mp.Process(target=process2, args=(1, data, 11), name="2"),
    ]

    info("Starting Processes")
    for p in processes:
        p.start()

    for i, p in enumerate(processes):
        p.join()
        info("Terminated process {}".format(p.name))
    info("Terminate: Done!")

    pass


if __name__ == "__main__":
    main()
