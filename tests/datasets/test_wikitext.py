from unittest import TestCase
from unittest import mock

from torchpipe.datasets import WikiText2, WikiText103


class WikiText2TestCase(TestCase):

    @mock.patch('torchpipe.datasets.wikitext.cached_get_wikitext')
    def test_dunder_init(self, cached_get_wikitext):
        WikiText2()
        cached_get_wikitext.called_once_with('wikitext-2')


class WikiText103TestCase(TestCase):

    @mock.patch('torchpipe.datasets.wikitext.cached_get_wikitext')
    def test_dunder_init(self, cached_get_wikitext):
        WikiText103()
        cached_get_wikitext.called_once_with('wikitext-103')
