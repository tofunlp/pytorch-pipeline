# PyTorch Pipeline: Simple ETL Pipeline for PyTorch

PyTorch Pipeline is a simple ETL framework for PyTorch.
It is an alternative to [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) in TensorFlow


# Requirements

- Python 3.6+
- PyTorch 1.2+


# Installation

To install PyTorch Pipeline:

```bash
pip install pytorch_pipeilne
```


# Basic Usage

```py
import pytorch_pipeilne as pp

d = pp.TextDataset('/path/to/your/text')
d.shuffle(buffer_size=100).batch(batch_size=10).first()
```

# Usage with PyTorch

```py
from torch.utils.data import DataLoader
import pytorch_pipeilne as pp


d = pp.Dataset(range(1_000)).parallel().shuffle(100).batch(10)
loader = DataLoader(d, num_workers=4, collate_fn=lambda x: x)
for x in loader:
    ...
```

# Usage with LineFlow

You can use PyTorch Pipeline with pre-defined datasets in [LineFlow](https://github.com/tofunlp/lineflow):

```py
from torch.utils.data import DataLoader
from lineflow.datasets.wikitext import cached_get_wikitext
import pytorch_pipeilne as pp

dataset = cached_get_wikitext('wikitext-2')
# Preprocessing dataset
train_data = pp.Dataset(dataset['train']) \
    .flat_map(lambda x: x.split() + ['<eos>']) \
    .window(35) \
    .parallel() \
    .shuffle(64 * 100) \
    .batch(64)

# Iterating dataset
loader = DataLoader(train_data, num_workers=4, collate_fn=lambda x: x)
for x in loader:
    ...
```
