# Training function
def train_rnn(rnn, data, char_to_ix, ix_to_char, seq_length=25, num_iterations=100_000, print_every=1000):
    loss_history = []
    h_prev = np.zeros((rnn.hidden_size, 1))
    
    for iteration in range(num_iterations):
        # Create a mini-batch
        start_idx = iteration % (len(data) - seq_length - 1)
        inputs = data[start_idx:start_idx + seq_length]
        targets = data[start_idx + 1:start_idx + seq_length + 1]
        
        # One-hot encode inputs and targets
        inputs_encoded = encode_sequence(inputs, char_to_ix, rnn.vocab_size)
        targets_encoded = [char_to_ix[ch] for ch in targets]
        
        # Forward pass
        ps, hs = rnn.forward(inputs_encoded, h_prev)
        
        # Compute gradients
        gradients = rnn.compute_gradients(inputs_encoded, targets_encoded, hs, ps)
        
        # Backward pass (parameter update)
        rnn.backward_adagrad(gradients)
        
        # Loss computation
        loss = -np.sum([np.log(ps[t][targets_encoded[t], 0]) for t in range(len(targets_encoded))])
        loss_history.append(loss)
        
        # Print loss and generate sample text periodically
        if iteration % print_every == 0:
            print(f"Iteration {iteration}, Loss: {loss}")
            seed_char = data[start_idx]
            sample = sample_text(rnn, seed_char, char_to_ix, ix_to_char)
            print(f"Sample Text:\n{sample}\n")

    # Plot the loss history
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()