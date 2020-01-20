from typing import Iterator
import os
import io

from pytorch_pipeline import Dataset


class TextDataset(Dataset):
    def __init__(self, path: str, encoding: str = 'utf-8') -> None:
        assert os.path.exists(path)

        self._path = path
        self._encoding = encoding

    def __iter__(self) -> Iterator[str]:
        with io.open(self._path, 'r', encoding=self._encoding) as fp:
            for line in fp:
                yield line.rstrip(os.linesep)
