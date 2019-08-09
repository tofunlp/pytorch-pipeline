from unittest import TestCase
from unittest import mock

from torchpipe.datasets import PennTreebank


class PennTreebankTestCase(TestCase):

    @mock.patch('torchpipe.datasets.penn_treebank.lfds')
    def test_dunder_init(self, lfds):
        lfds.get_penn_treebank.return_value = {'train': []}
        PennTreebank()
        lfds.get_penn_treebank.assert_called_once()
