from unittest import TestCase
from unittest import mock
import tempfile

import easyfile
from torchpipe.datasets import WikiText2, WikiText103


class WikiText2TestCase(TestCase):

    @mock.patch('torchpipe.datasets.wikitext.lfds')
    def test_dunder_init(self, lfds):
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(b'test\n')
            fp.seek(0)
            lfds.get_wikitext.return_value = {'train': easyfile.TextFile(fp.name)}
            WikiText2()
            lfds.get_wikitext.called_once_with('wikitext-2')


class WikiText103TestCase(TestCase):

    @mock.patch('torchpipe.datasets.wikitext.lfds')
    def test_dunder_init(self, lfds):
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(b'test\n')
            fp.seek(0)
            lfds.get_wikitext.return_value = {'train': easyfile.TextFile(fp.name)}
            WikiText103()
            lfds.get_wikitext.called_once_with('wikitext-103')
