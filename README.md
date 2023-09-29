# Memoria

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/cosmoquester/memoria/tree/master.svg?style=svg&circle-token=513f0f5e9a706a51509d198359fe0e016a227ce9)](https://dl.circleci.com/status-badge/redirect/gh/cosmoquester/memoria/tree/master)
[![codecov](https://codecov.io/gh/cosmoquester/memoria/branch/master/graph/badge.svg?token=KZdkgkBzZG)](https://codecov.io/gh/cosmoquester/memoria)

![Memoria-Engrams-compressed-1500](https://github.com/cosmoquester/memoria/assets/30718444/fa36dd13-7aac-4c4d-b749-83d93993d422)


Memoria is general memory network which can be used to memorize long sequence information.

Memoria is an independant module which can be applied to neural network models and the experiment code of paper is in the `experiment` directory.

## Installation

```sh
$ pip install git+ssh://git@github.com/cosmoquester/memoria
```

You can install memoria by pip command above.

## Tutorial

This is a tutorial to help to understand the concept and mechanism of Memoria.
Fake random data and lifespan delta are used for simplification.

```python
import torch
from memoria import Memoria, EngramType

torch.manual_seed(42)

# Memoria Parameters
num_reminded_stm = 4
stm_capacity = 16
ltm_search_depth = 5
initial_lifespan = 3
num_final_ltms = 4

# Data Parameters
batch_size = 2
sequence_length = 8
hidden_dim = 64

memoria = Memoria(
    num_reminded_stm=num_reminded_stm,
    stm_capacity=stm_capacity,
    ltm_search_depth=ltm_search_depth,
    initial_lifespan=initial_lifespan,
    num_final_ltms=num_final_ltms,
)
data = torch.rand(batch_size, sequence_length, hidden_dim)

# Add data as working memory
memoria.add_working_memory(data)

# Expected values
"""
>>> len(memoria.engrams)
16
>>> memoria.engrams.data.shape
torch.Size([2, 8, 64])
>>> memoria.engrams.lifespan
tensor([[3., 3., 3., 3., 3., 3., 3., 3.],
        [3., 3., 3., 3., 3., 3., 3., 3.]])
"""

reminded_memories, reminded_indices = memoria.remind()

# No reminded memories because there is no STM/LTM engrams yet
"""
>>> reminded_memories
tensor([], size=(2, 0, 64))
>>> reminded_indices
tensor([], size=(2, 0), dtype=torch.int64)
"""

memoria.adjust_lifespan_and_memories(reminded_indices, torch.zeros_like(reminded_indices))

# Decreases lifespan for all engrams & working memories have changed into shortterm memory
"""
>>> memoria.engrams.lifespan
tensor([[2., 2., 2., 2., 2., 2., 2., 2.],
        [2., 2., 2., 2., 2., 2., 2., 2.]])
>>> memoria.engrams.engrams_types
tensor([[2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.uint8)
>>> EngramType.SHORTTERM
<EngramType.SHORTTERM: 2>
"""

data2 = torch.rand(batch_size, sequence_length, hidden_dim)
memoria.add_working_memory(data2)

"""
>>> len(memoria.engrams)
32
>>> memoria.engrams.lifespan
tensor([[2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.],
        [2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.]])
"""

reminded_memories, reminded_indices = memoria.remind()

# Remind memories from STM
"""
>>> reminded_memories.shape
torch.Size([2, 6, 64])
>>> reminded_indices.shape
torch.Size([2, 6])
>>> reminded_indices
tensor([[ 0,  6,  4,  3,  2, -1],
        [ 0,  7,  6,  5,  4, -1]])
"""

# Increase lifespan of all the reminded engrams by 5
memoria.adjust_lifespan_and_memories(reminded_indices, torch.full_like(reminded_indices, 5))

# Reminded engrams got lifespan by 5, other engrams have got older
"""
>>> memoria.engrams.lifespan
>>> memoria.engrams.lifespan
tensor([[6., 1., 6., 6., 6., 1., 6., 1., 2., 2., 2., 2., 2., 2., 2., 2.],
        [6., 1., 1., 1., 6., 6., 6., 6., 2., 2., 2., 2., 2., 2., 2., 2.]])
"""

# This is default process to utilize Memoria
for _ in range(10):
    data = torch.rand(batch_size, sequence_length, hidden_dim)
    memoria.add_working_memory(data)

    reminded_memories, reminded_indices = memoria.remind()

    lifespan_delta = torch.randint_like(reminded_indices, 0, 6).float()

    memoria.adjust_lifespan_and_memories(reminded_indices, lifespan_delta)

# After 10 iteration, some engrams have changed into longterm memory and got large lifespan
# Engram type zero means those engrams are deleted
"""
>>> len(memoria.engrams)
72
>>> memoria.engrams.engrams_types
tensor([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], dtype=torch.uint8)
>>> EngramType.LONGTERM
<EngramType.LONGTERM: 3>
>>> EngramType.NULL
<EngramType.NULL: 0>
>>> memoria.engrams.lifespan
tensor([[ 9.,  1.,  8.,  2., 16.,  5., 13.,  7.,  7.,  3.,  3.,  4.,  3.,  3.,
          4.,  2.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  6.,  1.,  1.,
          2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
        [-1., -1.,  3.,  2., 19., 21., 11.,  6., 14.,  1.,  5.,  1.,  5.,  1.,
          5.,  1.,  1.,  8.,  2.,  1.,  1.,  1.,  2.,  1.,  1.,  1.,  1.,  1.,
          2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.]])
"""
```
