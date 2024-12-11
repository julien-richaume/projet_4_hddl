class RNN:
    # Simple "vanilla" RNN model

    def __init__(self, vocab_size, hidden_size=64):
        # Store the model hyperparameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Initialize the model weights and biases
        self.Waa = np.random.randn(hidden_size, hidden_size) / 1000
        self.Wax = np.random.randn(hidden_size, vocab_size) / 1000
        self.Wya = np.random.randn(vocab_size, hidden_size) / 1000
        self.ba = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
        # AdaGrad optimization parameters
        self.m = {k: np.zeros_like(v) for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        

    def forward(self, inputs, h_prev):
        """
        Perform forward pass.
        Args:
            inputs: One-hot encoded inputs of shape (vocab_size, seq_length)
            h_prev: Initial hidden state of shape (hidden_size, 1)
        Returns:
            outputs: List of output probabilities
            hs: List of hidden states
        """
        # Initialize hidden state, logits, and output pobabilities
        hs, ys, ps = {}, {}, {}
        hs[-1] = np.copy(h_prev)

        for t, x in enumerate(inputs):
            # Update the hidden state
            hs[t] = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, hs[t-1]) + self.ba)

            # Compute the output logits
            ys[t] = np.dot(self.Wya, hs[t]) + self.by

            # Softmax probabilities
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

        return ps, hs
    

    def compute_gradients(self, inputs, targets, hs, ps):
        """
        Compute the gradients for one sequence.
        Args:
            inputs: List of one-hot encoded inputs
            targets: List of integer targets
            hs: List of hidden states
            ps: List of output probabilities
        Returns:
            Gradients
        """
        # Initialize gradients
        dWax, dWaa, dWya = np.zeros_like(self.Wax), np.zeros_like(self.Waa), np.zeros_like(self.Wya)
        dba, dby = np.zeros_like(self.ba), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        # Loop through each time step
        for t in reversed(range(len(targets))):
            # Compute dy, the derivative of the loss with respect to the output
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            # Compute the gradient of the output layer
            dWya += np.dot(dy, hs[t].T)
            dby += dy

            # Backpropagate the gradient to the hidden layer
            dh = np.dot(self.Wya.T, dy) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh # Backprop through tanh

            # Compute the gradients of the hidden layer
            dWax += np.dot(dh_raw, inputs[t].T)
            dWaa += np.dot(dh_raw, hs[t].T)
            dba += dh_raw

            # Update the next hidden state
            dh_next = np.dot(self.Waa.T, dh_raw)

        # Clip gradients to mitigate exploding gradients
        for dparam in [dWax, dWaa, dWya, dba, dby]:
            np.clip(dparam, -1, 1, out=dparam)
        
        return dWax, dWaa, dWya, dba, dby


    def backward_adagrad(self, gradients, lr=0.1):
        for param, dparam in zip(['Wax', 'Waa', 'Wya', 'ba', 'by'], gradients):
            self.m[param] += dparam ** 2
            self.__dict__[param] -= lr * dparam / np.sqrt(self.m[param] + 1e-8)


    def sample(self, seed_char, char_to_ix, ix_to_char, n=50):
        """
        Generate text by sampling from the model.
        Args:
            seed_char: Seed character
            char_to_ix: Dictionary mapping characters to integers
            ix_to_char: Dictionary mapping integers to characters
            n: Number of characters to sample

        Returns:
            Sampled text
        """
        x = np.zeros((self.vocab_size, 1))
        x[char_to_ix[seed_char]] = 1
        h = np.zeros((self.hidden_size, 1))
        output = []

        for t in range(n):
            h = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, h) + self.ba)
            y = np.dot(self.Wya, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            output.append(ix_to_char[idx])
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

        return ''.join(output)