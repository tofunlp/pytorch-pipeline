from unittest import TestCase
from unittest import mock

from torchpipe.datasets import PennTreebank


class PennTreebankTestCase(TestCase):

    @mock.patch('torchpipe.datasets.penn_treebank.cached_get_penn_treebank')
    def test_dunder_init(self, cached_get_penn_treebank):
        PennTreebank()
        cached_get_penn_treebank.assert_called_once()
