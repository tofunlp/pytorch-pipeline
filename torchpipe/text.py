from typing import Iterator, List
import os
import io
import itertools

from torch.utils.data import get_worker_info

from torchpipe import Dataset


class TextDataset(Dataset):
    def __init__(self, path: str, encoding: str = 'utf-8') -> None:
        assert os.path.exists(path)

        self._path = path
        self._encoding = encoding

    def __iter__(self) -> Iterator[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            with io.open(self._path, 'r', encoding=self._encoding) as fp:
                for line in fp:
                    yield line.rstrip(os.linesep)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            with io.open(self._path, 'rb') as fp:
                for i, line in enumerate(fp):
                    if i % num_workers == worker_id:
                        yield line.decode(self._encoding).rstrip(os.linesep)


class ConcatTextDataset(Dataset):
    def __init__(self, paths: List[str], encoding: str = 'utf-8') -> None:
        assert all(os.path.exists(p) for p in paths)

        self._paths = paths
        self._encoding = encoding

    def __iter__(self) -> Iterator[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            fps = [io.open(p, 'r', encoding=self._encoding) for p in self._paths]
            for line in itertools.chain(*fps):
                yield line.rstrip(os.linesep)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            fps = [io.open(p, 'rb') for p in self._paths]
            for i, line in enumerate(itertools.chain(*fps)):
                if i % num_workers == worker_id:
                    yield line.decode(self._encoding).rstrip(os.linesep)
        for fp in fps:
            fp.close()


class ZipTextDataset(Dataset):
    def __init__(self, paths: List[str], encoding: str = 'utf-8') -> None:
        assert all(os.path.exists(p) for p in paths)

        self._paths = paths
        self._encoding = encoding

    def __iter__(self) -> Iterator[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            fps = [io.open(p, 'r', encoding=self._encoding) for p in self._paths]
            for lines in zip(*fps):
                yield tuple(line.rstrip(os.linesep) for line in lines)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            fps = [io.open(p, 'rb') for p in self._paths]
            for i, lines in enumerate(zip(*fps)):
                if i % num_workers == worker_id:
                    yield tuple(line.decode(self._encoding).rstrip(os.linesep)
                                for line in lines)
        for fp in fps:
            fp.close()
