#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyper Parameters
BATCH_SIZE = 32  # How many independent sequences will we process in parallel?
BLOCK_SIZE = 8  # What is the maximum context length for predictions?
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
# elif torch.backends.mps.is_available():
# I'm doing something wrong, my macbook shows mps is available
# but when using it, the training becomes very slow
# DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
EVAL_ITERS = 200

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    TEXT = f.read()

# Here are all the unique characters that occur in the text
UNIQUE_CHARS = sorted(set(TEXT))
VOCAB_SIZE = len(UNIQUE_CHARS)

# Create a mapping from character to integer and vice versa
STR_TO_INT = {char: i for i, char in enumerate(UNIQUE_CHARS)}
INT_TO_STR = {i: char for i, char in enumerate(UNIQUE_CHARS)}


def encode(text):
    """Convert given text characters to a list of integers using the str_to_int mapping."""
    return [STR_TO_INT[char] for char in text]


def decode(text):
    """Convert given list of integers to text using the int_to_str mapping."""
    return "".join([INT_TO_STR[i] for i in text])


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    """Retrieve a single random batch of samples/sequences.

    :param data: The data to sample from.
    :param batch_size: The number of independent sequences to process in parallel.
    :param block_size: The maximum context length (size of a single sample) for predictions.

    :return: A tuple containing the input and target batches.
    """
    random_data_idx = torch.randint(len(data) - block_size, (batch_size,))

    input_batch = torch.stack([data[i : i + block_size] for i in random_data_idx])
    target_batch = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in random_data_idx]
    )

    # Move the data to the device if we're using cuda
    input_batch = input_batch.to(DEVICE)
    target_batch = target_batch.to(DEVICE)

    return input_batch, target_batch


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Create token embedding table.
        # It will basically be a matrix of size (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx and targets are both (B, T) tensors of integers

        # We retrieve the token embeddings by index.
        # An embedding is 'Batch by Time by Channel'
        # In this case batch size is 4, time is 8 and channel is vocal_size=65
        # Logits are basically the scores for the next character in the sequence.
        logits: torch.Tensor = self.token_embedding_table(idx)  # (Batch, Time, Channel)

        # We want to calculate the loss for the model.
        # Basically we are asking the question: How well did the model predict the
        # targets, based on the logits.
        if targets is None:
            loss = None
        else:
            # NOTE: cross_entropy expects Batch*Time by Channel
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)
            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """Generate model predictions for a given input sequence.

        :param idx: The current index for a batch (Batch by Time)
        :param max_new_tokens: The maximum number of new tokens to generate
        """

        for _ in range(max_new_tokens):
            # Get the predictions
            # NOTE: self(idx) will end up calling the self.forward method
            logits, _ = self(idx)

            # Focus only on the last time step.
            logits = logits[:, -1, :]  # becomes (Batch, Channel)

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)  # (Batch, Channel)

            # Sample from the distribution (probabilities)
            # We will get a batch by 1 tensor of integers where each batch will
            # have a single prediction for the next token
            idx_next = torch.multinomial(probabilities, num_samples=1)  # (Batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (Batch, Time+1)

        return idx


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Estimate the loss of the model on the validation set.

    Calculates the mean (the average) of the loss over EVAL_ITERS iterations.

    torch.no_grad() tells pyTorch we won't do any backpropagation so it can be a lot
    more efficient in terms of memory and speed.

    :param model: The model to evaluate.
    :param train_data: The training data set.
    :param val_data: The validation data set.

    :return: A dictionary containing the training and validation loss.
    """
    output: dict[str, torch.Tensor] = {}

    # Set model to evaluation mode
    model.eval()

    # Helper function to calculate the loss
    def calculate_loss(data: torch.Tensor):
        """Calculate the loss given the data set."""
        losses = torch.zeros(EVAL_ITERS)
        for idx in range(EVAL_ITERS):
            input_batch, target_batch = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
            _, loss = model(input_batch, target_batch)
            losses[idx] = loss.item()

        return losses.mean()

    # Calculate the loss on the training and validation sets
    output["train_loss"] = calculate_loss(train_data)
    output["val_loss"] = calculate_loss(val_data)

    # Reset the model back to training mode
    model.train()

    return output


if __name__ == "__main__":
    """Script entrypoint."""

    # We will split up our data into training (90%) and validation sets (10%)
    print("Encoding data, initializing model and optimizer...")
    data = torch.tensor(encode(TEXT), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    # Initialize model
    model = BigramLanguageModel(vocab_size=VOCAB_SIZE)
    m = model.to(DEVICE)

    # Create pytorch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print("Training model...")
    for iter in range(MAX_ITERS):

        # Every once in a while evaluate the loss on train and val sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(
                f"Iteration {iter} | Train Loss: {losses['train_loss']:.4f} | Val Loss: {losses['val_loss']:.4f}"
            )

        # Sample a batch of data
        input_batch, target_batch = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)

        # Evaluate the loss
        logits, loss = model(input_batch, target_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Training complete! (Loss: {loss.item():.4f})\n\n")
    # Generate some text from the model

    # Since it's a simple bigram model it will be pretty bad, but
    # it should be a lot better already than the random output we got before training.
    # The bigram model is only looking at the last token and trying to predict the next token.
    # what we want is to look at the entire context (so multiple tokens stringed together) and predict the next token.
    print("Generating new text...\n")
    context = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
    new_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    print(f"{decode(new_tokens)}")
