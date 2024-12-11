# Sample text during training
def sample_text(rnn, seed_char, char_to_ix, ix_to_char, length=200):
    return rnn.sample(seed_char, char_to_ix, ix_to_char, length)