# Convert a sequence of characters to one-hot encoded vectors
def encode_sequence(sequence, char_to_ix, vocab_size):
    return [one_hot_encode(char_to_ix[ch], vocab_size) for ch in sequence]