from lineflow.datasets.wikitext import cached_get_wikitext

from torchpipe import Dataset


class WikiText2(Dataset):
    def __init__(self, split='train') -> None:
        super(WikiText2, self).__init__(cached_get_wikitext('wikitext-2')[split])


class WikiText103(Dataset):
    def __init__(self, split='train') -> None:
        super(WikiText103, self).__init__(cached_get_wikitext('wikitext-103')[split])
