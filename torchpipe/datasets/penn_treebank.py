import lineflow.datasets as lfds

from torchpipe import Dataset


class PennTreebank(Dataset):
    def __init__(self, split='train') -> None:
        super(PennTreebank, self).__init__(lfds.get_penn_treebank()[split])
