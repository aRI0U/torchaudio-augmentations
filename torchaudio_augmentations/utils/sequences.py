from typing import Iterator, List

import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import PackedSequence


def pad_or_trim(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if length == tensor.size(-1):
        return tensor
    if length > tensor.size(-1):
        return pad(tensor, (0, length - tensor.size(-1)))
    return tensor[..., :length]


def unpack_sequence_it(packed: PackedSequence, lengths: torch.Tensor) -> Iterator[torch.Tensor]:
    sum_batch_sizes = pad(packed.batch_sizes.to(packed.data.device), (1, 0)).cumsum(dim=0)
    for seq_idx, seq_length in zip(packed.unsorted_indices, lengths):
        indices = sum_batch_sizes[:seq_length] + seq_idx
        yield packed.data[indices]


def unpack_sequence(packed: PackedSequence, lengths: torch.Tensor) -> List[torch.Tensor]:
    sum_batch_sizes = pad(packed.batch_sizes.to(packed.data.device), (1, 0)).cumsum(dim=0)
    sequences = []
    for seq_idx, seq_length in zip(packed.unsorted_indices, lengths):
        indices = sum_batch_sizes[:seq_length] + seq_idx
        sequences.append(packed.data[indices])
    return sequences


if __name__ == "__main__":
    import time

    from torch.nn.utils.rnn import pack_sequence
    from torch.nn.utils.rnn import unpack_sequence as naive_unpack_sequence

    device = torch.device("cuda:0")

    num_sequences = 2048
    lengths = torch.randint(1, 96, (num_sequences,))
    seqs = [torch.randint(10, (length, 48), device=device) for length in lengths]

    packed = pack_sequence(seqs, enforce_sorted=False)

    t0 = time.time()
    unpacked_mine = unpack_sequence(packed, lengths)
    t1 = time.time()
    print(t1 - t0)

    t0 = time.time()
    unpacked_naive = naive_unpack_sequence(packed)
    t1 = time.time()
    print(t1 - t0)

    # print(unpacked_naive)
    # print(unpacked_mine)

    for a, b in zip(unpacked_naive, unpacked_mine):
        torch.testing.assert_close(a, b)
    print("OK.")
