from typing import Any, Iterator, Iterable, Tuple, List, Callable
import _collections_abc
import math
import random
import itertools

import easyfile
import lineflow as lf
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info


class Dataset(IterableDataset):
    def __init__(self, dataset: Iterable[Any]) -> None:
        assert isinstance(dataset, _collections_abc.Iterable)

        if isinstance(dataset, _collections_abc.Sequence):
            self._get_iterator = self._get_iterator_from_sequence
        elif isinstance(dataset, easyfile.TextFile):
            self._get_iterator = self._get_iterator_from_easyfile
        else:
            self._get_iterator = self._get_iterator_from_iterable

        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        yield from self._get_iterator()

    def _get_iterator_from_iterable(self) -> Iterator[Any]:
        worker_info = get_worker_info()
        iterable, self._dataset = itertools.tee(self._dataset)
        if worker_info is None:
            yield from iterable
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            yield from itertools.islice(iterable, worker_id, None, num_workers)

    def _get_iterator_from_sequence(self) -> Iterator[Any]:
        worker_info = get_worker_info()
        if worker_info is None:
            yield from self._dataset
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            span = math.ceil(len(self._dataset) / num_workers)
            start = span * worker_id
            end = span * (worker_id + 1)
            yield from self._dataset[start:end]

    def _get_iterator_from_easyfile(self) -> Iterator[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            yield from self._dataset
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            span = math.ceil(len(self._dataset) / num_workers)
            start = span * worker_id
            end = span * (worker_id + 1)
            yield from self._dataset.iterate(start, end)

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

    def bucket(self, bucket_size: int) -> 'BucketDataset':
        return BucketDataset(self, bucket_size)

    def shuffle(self, buffer_size: int = None) -> 'ShuffleDataset':
        return ShuffleDataset(self, buffer_size)

    def sort(self, sort_key: Callable, buffer_size: int = None) -> 'SortDataset':
        return SortDataset(self, sort_key, buffer_size)

    def window(self, window_size: int, shift: int = None) -> 'WindowDataset':
        return WindowDataset(self, window_size, shift)

    def concat(self, *others: Tuple['Dataset']) -> 'ConcatDataset':
        return ConcatDataset(self, *others)

    __add__ = concat

    def zip(self, *others: Tuple['Dataset']) -> 'ZipDataset':
        return ZipDataset(self, *others)

    @staticmethod
    def range(*args: Tuple[int]) -> 'RangeDataset':
        return RangeDataset(*args)


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
        yield from lf.flat_map(self._map_func, self._dataset, lazy=True)


class FilterDataset(Dataset):
    def __init__(self, dataset: Dataset, predicate: Callable[[Any], bool]) -> None:
        assert isinstance(dataset, Dataset)
        assert callable(predicate)

        self._predicate = predicate
        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        yield from lf.filter(self._predicate, self._dataset, lazy=True)


class BucketDataset(Dataset):
    def __init__(self, dataset: Dataset, bucket_size: int) -> None:
        assert isinstance(dataset, Dataset)

        self._bucket_size = bucket_size
        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        bucket = []
        for x in self._dataset:
            bucket.append(x)
            if len(bucket) == self._bucket_size:
                yield bucket
                bucket = []
        if bucket:
            yield bucket


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
        else:
            for x in self._dataset:
                chunk.append(x)
                if len(chunk) == self._buffer_size:
                    random.shuffle(chunk)
                    yield from chunk
                    chunk = []
            if chunk:
                random.shuffle(chunk)
                yield from chunk


class SortDataset(Dataset):
    def __init__(self, dataset: Dataset, sort_key: Callable, buffer_size: int = None) -> None:
        assert isinstance(dataset, Dataset)
        assert callable(sort_key)

        self._dataset = dataset
        self._sort_key = sort_key
        self._buffer_size = buffer_size

    def __iter__(self) -> Iterator[Any]:
        if self._buffer_size is None:
            yield from sorted(self._dataset, key=self._sort_key)
        else:
            chunk = []
            for x in self._dataset:
                chunk.append(x)
                if len(chunk) == self._buffer_size:
                    chunk.sort(key=self._sort_key)
                    yield from chunk
                    chunk = []
            if chunk:
                chunk.sort(key=self._sort_key)
                yield from chunk


class RangeDataset(Dataset):
    def __init__(self, *args: Tuple[int]) -> None:
        self._args = args

    def __iter__(self) -> Iterator[int]:
        worker_info = get_worker_info()
        if worker_info is None:
            yield from range(*self._args)
        else:
            base = range(*self._args)
            start = base.start
            end = base.stop
            step = base.step
            yield from range(start + step * worker_info.id, end, step * worker_info.num_workers)


class WindowDataset(Dataset):
    def __init__(self, dataset: Dataset, window_size: int, shift: int = None) -> None:
        assert isinstance(dataset, Dataset)

        self._dataset = dataset
        self._window_size = window_size
        self._shift = shift or window_size

    def __iter__(self) -> Iterator[Any]:
        yield from lf.window(self._dataset, self._window_size, self._shift, lazy=True)


class ConcatDataset(Dataset):
    def __init__(self, dataset: Dataset, *others: Tuple[Dataset]) -> None:
        assert isinstance(dataset, Dataset)
        assert all(isinstance(d, Dataset) for d in others)

        self._dataset = dataset
        self._others = others

    def __iter__(self):
        yield from itertools.chain(self._dataset, *self._others)


class ZipDataset(Dataset):
    def __init__(self, dataset: Dataset, *others: Tuple[Dataset]) -> None:
        assert isinstance(dataset, Dataset)
        assert all(isinstance(d, Dataset) for d in others)

        self._dataset = dataset
        self._others = others

    def __iter__(self):
        yield from zip(self._dataset, *self._others)
