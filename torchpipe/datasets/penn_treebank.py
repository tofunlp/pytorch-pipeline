from lineflow.datasets.penn_treebank import cached_get_penn_treebank

from torchpipe import Dataset


class PennTreebank(Dataset):
    def __init__(self, split='train') -> None:
        super(PennTreebank, self).__init__(cached_get_penn_treebank()[split])
