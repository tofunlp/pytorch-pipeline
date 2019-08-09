import lineflow.datasets as lfds

from torchpipe.core import SequenceDataset


class PennTreebank(SequenceDataset):
    def __init__(self, split='train') -> None:
        super(PennTreebank, self).__init__(lfds.get_penn_treebank()[split])
