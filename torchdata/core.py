from typing import Any, Iterator, Iterable, List, Callable
import math
import random
import itertools
import collections

from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info


class Dataset(IterableDataset):
    def all(self) -> List[Any]:
        return list(self)

    def first(self) -> Any:
        return next(iter(self))

    def take(self, n) -> List[Any]:
        return list(itertools.islice(self, n))

    def apply(self,
              transformation_func: Callable[[Iterator[Any]], Iterator[Any]]
              ) -> 'ApplyDataset':
        return ApplyDataset(self, transformation_func)

    def map(self, map_func: Callable[[Any], Any]) -> 'MapDataset':
        return MapDataset(self, map_func)

    def flat_map(self, map_func: Callable[[Any], Iterable[Any]]) -> 'FlatMapDataset':
        return FlatMapDataset(self, map_func)

    def filter(self, predicate: Callable[[Any], bool]) -> 'FilterDataset':
        return FilterDataset(self, predicate)

    def shuffle(self, buffer_size: int = None) -> 'ShuffleDataset':
        return ShuffleDataset(self, buffer_size)

    def window(self, window_size: int, shift: int = None) -> 'WindowDataset':
        return WindowDataset(self, window_size, shift)

    def concat(self, *others: List['Dataset']) -> 'ConcatDataset':
        return ConcatDataset(self, *others)

    __add__ = concat

    def zip(self, *others: List['Dataset']) -> 'ZipDataset':
        return ZipDataset(self, *others)

    @staticmethod
    def range(n: int) -> 'RangeDataset':
        return RangeDataset(n)


class ApplyDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 transformation_func: Callable[[Iterator[Any]], Iterator[Any]]
                 ) -> None:
        assert isinstance(dataset, Dataset)
        assert callable(transformation_func)

        self._transformation_func = transformation_func
        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        return self._transformation_func(self._dataset)


class MapDataset(Dataset):
    def __init__(self, dataset: Dataset, map_func: Callable[[Any], Any]) -> None:
        assert isinstance(dataset, Dataset)
        assert callable(map_func)

        self._map_func = map_func
        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        yield from map(self._map_func, self._dataset)


class FlatMapDataset(MapDataset):
    def __iter__(self) -> Iterator[Any]:
        yield from itertools.chain.from_iterable(map(self._map_func, self._dataset))


class FilterDataset(Dataset):
    def __init__(self, dataset: Dataset, predicate: Callable[[Any], bool]) -> None:
        assert isinstance(dataset, Dataset)
        assert callable(predicate)

        self._predicate = predicate
        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        yield from filter(self._predicate, self._dataset)


class ShuffleDataset(Dataset):
    def __init__(self, dataset: Dataset, buffer_size: int = None) -> None:
        assert isinstance(dataset, Dataset)

        self._buffer_size = buffer_size
        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        chunk = []

        if self._buffer_size is None:
            for x in self._dataset:
                chunk.append(x)
            random.shuffle(chunk)
            yield from chunk
            return

        for x in self._dataset:
            chunk.append(x)
            if len(chunk) >= self._buffer_size:
                random.shuffle(chunk)
                yield from chunk
                chunk = []
        if chunk:
            random.shuffle(chunk)
            yield from chunk


class RangeDataset(Dataset):
    def __init__(self, n: int) -> None:
        self._n = n

    def __iter__(self) -> Iterator[int]:
        worker_info = get_worker_info()
        if worker_info is None:
            yield from range(self._n)
        else:
            per_worker = math.ceil(self._n / worker_info.num_workers)
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self._n)
            yield from range(start, end)


class WindowDataset(Dataset):
    def __init__(self, dataset: Dataset, window_size: int, shift: int = None) -> None:
        assert isinstance(dataset, Dataset)

        self._dataset = dataset
        self._window_size = window_size
        self._shift = shift or 1

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self._dataset)
        window = collections.deque([], self._window_size)
        append = window.append

        for _ in range(self._window_size):
            append(next(iterator))
        yield tuple(window)

        i = 0
        for x in iterator:
            append(x)
            i = (i + 1) % self._shift
            if i % self._shift == 0:
                yield tuple(window)

        if (i % self._shift) and (self._shift - i < self._window_size):
            popleft = window.popleft
            for _ in range(self._shift - i):
                popleft()
        yield tuple(window)


class ConcatDataset(Dataset):
    def __init__(self, dataset: Dataset, *others: List[Dataset]) -> None:
        assert isinstance(dataset, Dataset)
        assert all(isinstance(d, Dataset) for d in others)

        self._dataset = dataset
        self._others = others

    def __iter__(self):
        yield from itertools.chain(self._dataset, *self._others)


class ZipDataset(Dataset):
    def __init__(self, dataset: Dataset, *others: List[Dataset]) -> None:
        assert isinstance(dataset, Dataset)
        assert all(isinstance(d, Dataset) for d in others)

        self._dataset = dataset
        self._others = others

    def __iter__(self):
        yield from zip(self._dataset, *self._others)
