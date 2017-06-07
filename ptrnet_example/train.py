import random
import numpy as np

def generate_nested_sequence(length, min_seglen=2, max_seglen=4):
    """Generate low-high-low sequence, with indexes of the first/last high/middle elements"""

    # Low (1-5) vs. High (6-10)
    seq_before = [(random.randint(1,5)) for x in range(random.randint(min_seglen, max_seglen))]
    seq_during = [(random.randint(6,9)) for x in range(random.randint(min_seglen, max_seglen))]
    seq_after = [random.randint(1,5) for x in range(random.randint(min_seglen, max_seglen))]
    seq = seq_before + seq_during + seq_after

    # Pad it up to max len with 0's
    seq = seq + ([0] * (length - len(seq)))
    return [seq, len(seq_before), len(seq_before) + len(seq_during)-1]

def create_one_hot(length, index):
    """Returns 1 at the index positions; can be scaled by client"""
    a = np.zeros([length])
    a[index] = 1.0
    return a

def generate_sample(max_length, batch_size=32):
    sequences = []
    first_indexes = []
    second_indexes = []

    training_segment_lengths = (11, 20)

    # Note that our training/testing datasets are the same size as our batch. This is
    #   unusual and just makes the code slightly simpler. In general your dataset size
    #   is >> your batch size and you rotate batches from the dataset through.
    for batch_index in range(batch_size):
        data = generate_nested_sequence(max_length,
                                        training_segment_lengths[0],
                                        training_segment_lengths[1])
        sequences.append(data[0])                                         # J
        first_indexes.append(create_one_hot(max_length, data[1]))         # J
        second_indexes.append(create_one_hot(max_length, data[2]))        # J

    inputs = np.stack(sequences) # B x J
    targets = np.stack([np.stack(first_indexes), np.stack(second_indexes)]) # I x B x J


    return inputs, targets

def generate_trainset(num_batches, batch_size, maxlen):

    trainset = []

    for i in range(num_batches):
        inputs, targets = generate_sample(maxlen, batch_size)
        trainset.append((inputs, targets))

    return trainset


if __name__ == '__main__':
    trainset = generate_trainset(num_batches=2, 
            batch_size=3, maxlen=60)

    print('_______________')
    print(trainset)
    print('_______________')
