import numpy as np
from typing import List
from abc import ABC, abstractmethod


class CollectorBase(ABC):

    def __init__(self):
        self.reset()

    @abstractmethod
    def accept(self, data):
        pass

    @abstractmethod
    def reset(self):
        pass


class SaveAllCollector(CollectorBase):

    def accept(self, data):
        self.__storage.append(data)

    def to_np_array(self):
        return np.asarray(self.__storage)

    def reset(self):
        self.__storage = []


class SaveLastCollector(CollectorBase):

    def accept(self, data):
        self.__value = data

    def get_last(self):
        return self.__value

    def reset(self):
        self.__value = None


def accept_collectors(collestors: List[CollectorBase], value):
    if collestors is not None:
        for coll in collestors:
            coll.accept(value)


def reset_collectors(collestors: List[CollectorBase]):
    if collestors is not None:
        for coll in collestors:
            coll.reset()
