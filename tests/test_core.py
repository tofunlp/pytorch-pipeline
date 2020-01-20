from unittest import TestCase
from itertools import chain

from torch.utils.data import DataLoader

from pytorch_pipeline import Dataset


class DatasetTestCase(TestCase):

    def setUp(self):
        self.base = range(100)
        self.data = Dataset(self.base)

    def test_dunder_iter(self):
        dataset = Dataset(self.base)
        for _ in range(100):
            for x, y in zip(dataset, self.base):
                self.assertEqual(x, y)

    def test_iterates_with_dataloader(self):
        loader = DataLoader(self.data.parallel().shuffle(11),
                            batch_size=16,
                            num_workers=0,
                            collate_fn=lambda x: x)
        with self.assertWarns(RuntimeWarning):
            self.assertSequenceEqual(sorted(x for batch in loader for x in batch),
                                     self.base)

    def test_iterates_with_dataloader_in_parallel(self):
        loader = DataLoader(self.data.parallel().shuffle(11),
                            batch_size=16,
                            num_workers=2,
                            collate_fn=lambda x: x)
        self.assertSequenceEqual(sorted(x for batch in loader for x in batch),
                                 self.base)

    def test_all(self):
        self.assertListEqual(self.data.all(), list(self.base))

    def test_apply(self):
        def f(it):
            for x in it:
                if x % 2 == 0:
                    yield x ** 2

        expected = map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, self.base))

        for x, y in zip(self.data.apply(f), expected):
            self.assertEqual(x, y)

    def test_batch(self):
        for i, x in enumerate(self.data.batch(3)):
            self.assertEqual(x, list(self.base[i * 3: i * 3 + 3]))

    def test_concat(self):
        other = Dataset(range(100))
        for x, y in zip(self.data.concat(other), chain(self.base, self.base)):
            self.assertEqual(x, y)

    def test_flat_map(self):
        def f(x): return [x] * 5

        for x, y in zip(self.data.flat_map(f), chain.from_iterable(map(f, self.base))):
            self.assertEqual(x, y)

    def test_filter(self):
        def f(x): return x % 2 == 0

        for x, y in zip(self.data.filter(f), filter(f, self.base)):
            self.assertEqual(x, y)

    def test_first(self):
        self.assertEqual(self.data.first(), self.base[0])

    def test_map(self):
        def f(x): return x ** 2

        for x, y in zip(self.data.map(f), map(f, self.base)):
            self.assertEqual(x, y)

    def test_shard(self):
        n = 4
        for i in range(n):
            for x, y in zip(self.data.shard(n, i), self.base[i::n]):
                self.assertEqual(x, y)

    def test_shuffles_data_with_buffer(self):
        for x, y in zip(sorted(self.data.shuffle(3)), self.base):
            self.assertEqual(x, y)

    def test_shuffles_data_without_buffer(self):
        for x, y in zip(sorted(self.data.shuffle()), self.base):
            self.assertEqual(x, y)

    def test_sorts_data_with_buffer(self):
        expected = chain.from_iterable(reversed(self.base[i: i + 3])
                                       for i in range(0, len(self.base), 3))
        for x, y in zip(self.data.sort(lambda x: -x, 3),  expected):
            self.assertEqual(x, y)

    def test_sorts_data_without_buffer(self):
        for x, y in zip(self.data.sort(lambda x: -x),  reversed(self.base)):
            self.assertEqual(x, y)

    def test_take(self):
        n = 50
        self.assertListEqual(self.data.take(n), list(self.base[:n]))

    def test_window(self):
        for x, y in zip(chain.from_iterable(self.data.window(3, 3)), self.base):
            self.assertEqual(x, y)

    def test_zip(self):
        other = Dataset(range(100))
        for x, i in zip(self.data.zip(other), self.base):
            self.assertTupleEqual(x, (i, i))
