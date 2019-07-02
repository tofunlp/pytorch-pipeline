from unittest import TestCase
from itertools import chain

from torchpipe import Dataset


class DatasetTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.data = Dataset.range(100)

    def test_apply(self):
        def f(it):
            for x in it:
                if x % 2 == 0:
                    yield x ** 2

        expected = map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, self.base))

        for x, y in zip(self.data.apply(f), expected):
            self.assertEqual(x, y)

    def test_map(self):
        def f(x): return x ** 2

        for x, y in zip(self.data.map(f), map(f, self.base)):
            self.assertEqual(x, y)

    def test_filter(self):
        def f(x): return x % 2 == 0

        for x, y in zip(self.data.filter(f), filter(f, self.base)):
            self.assertEqual(x, y)

    def test_flat_map(self):
        def f(x): return [x] * 5

        for x, y in zip(self.data.flat_map(f), chain.from_iterable(map(f, self.base))):
            self.assertEqual(x, y)

    def test_shuffles_data_with_buffer(self):
        for x, y in zip(sorted(self.data.shuffle(3)), self.base):
            self.assertEqual(x, y)

    def test_shuffles_data_without_buffer(self):
        for x, y in zip(sorted(self.data.shuffle()), self.base):
            self.assertEqual(x, y)

    def test_all(self):
        self.assertListEqual(self.data.all(), list(self.base))

    def test_first(self):
        self.assertEqual(self.data.first(), self.base[0])

    def test_take(self):
        n = 50
        self.assertListEqual(self.data.take(n), list(self.base[:n]))

    def test_window(self):
        for x, y in zip(chain.from_iterable(self.data.window(3, 3)), self.base):
            self.assertEqual(x, y)

    def test_zip(self):
        other = Dataset.range(100)
        for x, i in zip(self.data.zip(other), self.base):
            self.assertTupleEqual(x, (i, i))

    def test_concat(self):
        other = Dataset.range(100)
        for x, y in zip(self.data.concat(other), chain(self.base, self.base)):
            self.assertEqual(x, y)


class RangeDatasetTestCase(TestCase):

    def setUp(self):
        self.end = 100

    def test_dunder_init(self):
        d1 = Dataset.range(self.end)
        d2 = Dataset.range(50, self.end)
        d3 = Dataset.range(0, self.end, 3)
        self.assertTupleEqual(d1._args, (self.end,))
        self.assertTupleEqual(d2._args, (50, self.end))
        self.assertTupleEqual(d3._args, (0, self.end, 3))

    def test_dunder_iter(self):
        d1 = Dataset.range(self.end)
        d2 = Dataset.range(50, self.end)
        d3 = Dataset.range(0, self.end, 3)
        self.assertListEqual(d1.all(), list(range(self.end)))
        self.assertListEqual(d2.all(), list(range(50, self.end)))
        self.assertListEqual(d3.all(), list(range(0, self.end, 3)))

    def test_range_with_pytorch_dataloader(self):
        from torch.utils.data import DataLoader

        def get_loader(dataset):
            return DataLoader(dataset,
                              batch_size=16,
                              collate_fn=lambda x: x,
                              shuffle=False,
                              num_workers=2)

        d1 = Dataset.range(self.end)
        d2 = Dataset.range(50, self.end)
        d3 = Dataset.range(0, self.end, 3)
        loader = get_loader(d1)
        self.assertListEqual(
            list(sorted(chain.from_iterable(loader))),
            list(range(self.end))
        )
        loader = get_loader(d2)
        self.assertListEqual(
            list(sorted(chain.from_iterable(loader))),
            list(range(50, self.end))
        )
        loader = get_loader(d3)
        self.assertListEqual(
            list(sorted(chain.from_iterable(loader))),
            list(range(0, self.end, 3))
        )
