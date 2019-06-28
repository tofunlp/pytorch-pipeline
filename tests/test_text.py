from unittest import TestCase
import tempfile
from itertools import chain

from torch.utils.data import DataLoader

from torchpipe import TextDataset, ConcatTextDataset, ZipTextDataset


class TextDatasetTestCase(TestCase):

    def setUp(self):
        self.fp = tempfile.NamedTemporaryFile()
        self.n = 100
        for i in range(self.n):
            self.fp.write(f'line #{str(i).zfill(3)}\n'.encode('utf-8'))
        self.fp.seek(0)
        self.data = TextDataset(self.fp.name)

    def tearDown(self):
        self.fp.close()

    def test_dunder_init(self):
        self.assertEqual(self.data._path, self.fp.name)

    def test_dunder_iter(self):
        for x, i in zip(self.data, range(self.n)):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')

    def test_loads_with_torch_dataloader(self):
        loader = DataLoader(self.data,
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        for x, i in zip(sorted(chain.from_iterable(loader)), range(self.n)):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')

    def test_loads_with_torch_dataloader_after_shuffle(self):
        loader = DataLoader(self.data.shuffle(16 * 2),
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        for x, i in zip(sorted(chain.from_iterable(loader)), range(self.n)):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')


class ConcatTextDatasetTestCase(TestCase):

    def setUp(self):
        self.fp = tempfile.NamedTemporaryFile()
        self.n = 100
        for i in range(self.n):
            self.fp.write(f'line #{str(i).zfill(3)}\n'.encode('utf-8'))
        self.fp.seek(0)
        self.data = ConcatTextDataset([self.fp.name, self.fp.name])

    def tearDown(self):
        self.fp.close()

    def test_dunder_init(self):
        self.assertListEqual(self.data._paths, [self.fp.name, self.fp.name])

    def test_dunder_iter(self):
        for x, i in zip(self.data, chain(range(self.n), range(self.n))):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')

    def test_loads_with_torch_dataloader(self):
        loader = DataLoader(self.data,
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        expected = chain.from_iterable(map(lambda x: [x, x], range(self.n)))
        for x, i in zip(sorted(chain.from_iterable(loader)), expected):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')

    def test_loads_with_torch_dataloader_after_shuffle(self):
        loader = DataLoader(self.data.shuffle(16 * 2),
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        expected = chain.from_iterable(map(lambda x: [x, x], range(self.n)))
        for x, i in zip(sorted(chain.from_iterable(loader)), expected):
            self.assertEqual(x, f'line #{str(i).zfill(3)}')


class ZipTextDatasetTestCase(TestCase):

    def setUp(self):
        self.fp = tempfile.NamedTemporaryFile()
        self.n = 100
        for i in range(self.n):
            self.fp.write(f'line #{str(i).zfill(3)}\n'.encode('utf-8'))
        self.fp.seek(0)
        self.data = ZipTextDataset([self.fp.name, self.fp.name])

    def tearDown(self):
        self.fp.close()

    def test_dunder_init(self):
        self.assertListEqual(self.data._paths, [self.fp.name, self.fp.name])

    def test_dunder_iter(self):
        for x, i in zip(self.data, range(self.n)):
            line = f'line #{str(i).zfill(3)}'
            self.assertTupleEqual(x, (line, line))

    def test_loads_with_torch_dataloader(self):
        loader = DataLoader(self.data,
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        for x, i in zip(sorted(chain.from_iterable(loader)), range(self.n)):
            line = f'line #{str(i).zfill(3)}'
            self.assertTupleEqual(x, (line, line))

    def test_loads_with_torch_dataloader_after_shuffle(self):
        loader = DataLoader(self.data.shuffle(16 * 2),
                            batch_size=16,
                            collate_fn=lambda x: x,
                            shuffle=False,
                            num_workers=2)
        for x, i in zip(sorted(chain.from_iterable(loader)), range(self.n)):
            line = f'line #{str(i).zfill(3)}'
            self.assertTupleEqual(x, (line, line))
