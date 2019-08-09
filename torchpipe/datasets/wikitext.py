import lineflow.datasets as lfds
from torchpipe.core import EasyfileDataset


class WikiText2(EasyfileDataset):
    def __init__(self, split='train') -> None:
        super(WikiText2, self).__init__(lfds.get_wikitext('wikitext-2')[split])


class WikiText103(EasyfileDataset):
    def __init__(self, split='train') -> None:
        super(WikiText103, self).__init__(lfds.get_wikitext('wikitext-103')[split])
