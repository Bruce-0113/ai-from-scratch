"""Text classification models: a CNN with multi-width convolutions and a
bidirectional LSTM, plus a small simulation illustrating vanishing gradients
in recurrent networks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """CNN for text classification using parallel 1D convolutions over multiple filter widths.

    Each filter width acts as an n-gram detector; outputs are max-pooled over
    the sequence and concatenated before the final classification layer.
    """

    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            embed_dim: Dimensionality of token embeddings.
            n_classes: Number of output classes.
            filter_widths: Kernel sizes (n-gram widths) for the parallel conv branches.
            n_filters: Number of output channels per conv branch.
            dropout: Dropout probability applied before the final linear layer.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        """
        Args:
            token_ids: LongTensor of shape (batch, seq_len).

        Returns:
            Tensor of shape (batch, n_classes) with unnormalized class scores.
        """
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))


class LSTMClassifier(nn.Module):
    """Text classifier built on a (optionally bidirectional) LSTM with max-pooling over time."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            embed_dim: Dimensionality of token embeddings.
            hidden_dim: LSTM hidden state size (per direction).
            n_classes: Number of output classes.
            bidirectional: Whether to run the LSTM in both directions.
            dropout: Dropout probability applied before the final linear layer.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        """
        Args:
            token_ids: LongTensor of shape (batch, seq_len).

        Returns:
            Tensor of shape (batch, n_classes) with unnormalized class scores.
        """
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))


def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    """Simulate the magnitude of a gradient signal after backpropagating through time.

    Models the gradient contribution decaying geometrically as
    recurrent_weight ** seq_len, illustrating why vanilla RNNs struggle to
    learn long-range dependencies.

    Args:
        seq_len: Number of time steps the gradient is propagated through.
        recurrent_weight: Per-step multiplicative factor (< 1 causes vanishing,
            > 1 would cause exploding gradients).

    Returns:
        The simulated gradient magnitude after seq_len steps.
    """
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
