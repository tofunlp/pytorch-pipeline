from typing import Any, Iterator, Iterable, Tuple, List, Callable
import warnings
import _collections_abc
import random
import itertools

import lineflow as lf
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info


class Dataset(IterableDataset):
    def __init__(self, dataset: Iterable[Any]) -> None:
        assert isinstance(dataset, _collections_abc.Iterable)

        self._dataset = dataset

    def __iter__(self) -> Iterator[Any]:
        iterable, self._dataset = itertools.tee(self._dataset)
        yield from iterable

    def all(self) -> List[Any]:
        return list(self)

    def apply(self,
              transformation_func: Callable[[Iterator[Any]], Iterator[Any]]
              ) -> 'ApplyDataset':
        return ApplyDataset(self, transformation_func)

    def batch(self, batch_size: int) -> 'BatchDataset':
        return BatchDataset(self, batch_size)

    def concat(self, *others: Tuple['Dataset']) -> 'ConcatDataset':
        return ConcatDataset(self, *others)

    def flat_map(self, map_func: Callable[[Any], Iterable[Any]]) -> 'FlatMapDataset':
        return FlatMapDataset(self, map_func)

    def filter(self, predicate: Callable[[Any], bool]) -> 'FilterDataset':
        return FilterDataset(self, predicate)

    def first(self) -> Any:
        return next(iter(self))

    def map(self, map_func: Callable[[Any], Any]) -> 'MapDataset':
        return MapDataset(self, map_func)

    def parallel(self) -> 'ParallelDataset':
        return ParallelDataset(self)

    def shard(self, num_shards, index) -> 'ShardDataset':
        return ShardDataset(self, num_shards, index)

    def shuffle(self, buffer_size: int = None) -> 'ShuffleDataset':
        return ShuffleDataset(self, buffer_size)

    def sort(self, sort_key: Callable, buffer_size: int = None) -> 'SortDataset':
        return SortDataset(self, sort_key, buffer_size)

    def take(self, n) -> List[Any]:
        return list(itertools.islice(self, n))

    def window(self, window_size: int, shift: int = None) -> 'WindowDataset':
        return WindowDataset(self, window_size, shift)

    def zip(self, *others: Tuple['Dataset']) -> 'ZipDataset':
        return ZipDataset(self, *others)

    __add__ = concat


class ApplyDataset(Dataset):
    def __init__(self,
                 dataset: Dataset,
                 transformation_func: Callable[[Iterator[Any]], Iterator[Any]]
                 ) -> None:
        super(ApplyDataset, self).__init__(dataset)

        assert callable(transformation_func)

        self._transformation_func = transformation_func

    def __iter__(self) -> Iterator[Any]:
        return self._transformation_func(self._dataset)


class BatchDataset(Dataset):
    def __init__(self, dataset: Dataset, batch_size: int) -> None:
        super(BatchDataset, self).__init__(dataset)

        self._batch_size = batch_size

    def __iter__(self) -> Iterator[Any]:
        batch = []
        for x in self._dataset:
            batch.append(x)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


class ConcatDataset(Dataset):
    def __init__(self, dataset: Dataset, *others: Tuple[Dataset]) -> None:
        super(ConcatDataset, self).__init__(dataset)

        assert all(isinstance(d, Dataset) for d in others)

        self._others = others

    def __iter__(self):
        yield from itertools.chain(self._dataset, *self._others)


class FlatMapDataset(Dataset):
    def __init__(self, dataset: Dataset, map_func: Callable[[Any], Iterable[Any]]) -> None:
        super(FlatMapDataset, self).__init__(dataset)

        assert callable(map_func)

        self._map_func = map_func

    def __iter__(self) -> Iterator[Any]:
        yield from lf.flat_map(self._map_func, self._dataset, lazy=True)


class FilterDataset(Dataset):
    def __init__(self, dataset: Dataset, predicate: Callable[[Any], bool]) -> None:
        super(FilterDataset, self).__init__(dataset)

        assert callable(predicate)

        self._predicate = predicate

    def __iter__(self) -> Iterator[Any]:
        yield from filter(self._predicate, self._dataset)


class MapDataset(Dataset):
    def __init__(self, dataset: Dataset, map_func: Callable[[Any], Any]) -> None:
        super(MapDataset, self).__init__(dataset)

        assert callable(map_func)

        self._map_func = map_func

    def __iter__(self) -> Iterator[Any]:
        yield from map(self._map_func, self._dataset)


class ParallelDataset(Dataset):
    def __iter__(self) -> Iterator[Any]:
        worker_info = get_worker_info()
        if worker_info is None:
            warnings.warn(
                'Parallel is not activated. Please refer to '
                'torch.utils.data.DataLoader.',
                RuntimeWarning,
                stacklevel=2
            )
            yield from self._dataset
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            yield from self._dataset.shard(num_workers, worker_id)


class ShardDataset(Dataset):
    def __init__(self, dataset: Dataset, num_shards: int, index: int) -> None:
        super(ShardDataset, self).__init__(dataset)

        self._num_shards = num_shards
        self._index = index

    def __iter__(self) -> Iterator[Any]:
        yield from itertools.islice(self._dataset, self._index, None, self._num_shards)


class ShuffleDataset(Dataset):
    def __init__(self, dataset: Dataset, buffer_size: int = None) -> None:
        super(ShuffleDataset, self).__init__(dataset)

        self._buffer_size = buffer_size

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
        super(SortDataset, self).__init__(dataset)

        assert callable(sort_key)

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


class WindowDataset(Dataset):
    def __init__(self, dataset: Dataset, window_size: int, shift: int = None) -> None:
        super(WindowDataset, self).__init__(dataset)

        self._window_size = window_size
        self._shift = shift or window_size

    def __iter__(self) -> Iterator[Any]:
        yield from lf.window(self._dataset, self._window_size, self._shift, lazy=True)


class ZipDataset(Dataset):
    def __init__(self, dataset: Dataset, *others: Tuple[Dataset]) -> None:
        super(ZipDataset, self).__init__(dataset)

        assert all(isinstance(d, Dataset) for d in others)

        self._others = others

    def __iter__(self):
        yield from zip(self._dataset, *self._others)
