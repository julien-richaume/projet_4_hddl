# One-hot encode a character
def one_hot_encode(idx, vocab_size):
    one_hot = np.zeros((vocab_size, 1))
    one_hot[idx] = 1
    return one_hot