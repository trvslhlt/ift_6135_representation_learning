import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # weights for input
        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        # biases for input
        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        # weights for hidden state
        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # biases for hidden state
        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)


    def forward(
            self, 
            inputs: torch.FloatTensor, 
            hidden_states: torch.FloatTensor
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """GRU.

        This is a Gated Recurrent Unit
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
          The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The (initial) hidden state.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
          A feature tensor encoding the input sentence.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The final hidden state.
        """
        _, sequence_length, _ = inputs.shape
        h_t = hidden_states.squeeze(0)

        outputs = []
        
        # F.sigmoid is deprecated, use torch.sigmoid
        # T transposes all dimensions, t() raises an error if there are more than 2 dimensions
        for i in range(sequence_length):
            x_t = inputs[:, i, :] # select all features from all batches at time i
            # rt = σ(xtWTir + bir + ht−1WThr + bhr)    
            r_t = torch.sigmoid(x_t @ self.w_ir.t() + self.b_ir + h_t @ self.w_hr.t() + self.b_hr)
            # zt = σ(xtWTiz + biz + ht−1WThz + bhz)
            z_t = torch.sigmoid(x_t @ self.w_iz.t() + self.b_iz + h_t @ self.w_hz.t() + self.b_hz)
            # nt = tanh(xtWTin + bin + rt ∗(ht−1WThi + bhi))
            n_t = torch.tanh(x_t @ self.w_in.t() + self.b_in + r_t * (h_t @ self.w_hn.t() + self.b_hn))
            # ht = (1−zt) ∗nt + zt ∗ht−1
            h_t = (1 - z_t) * n_t + z_t * h_t
            outputs.append(h_t.unsqueeze(1)) # add a dimension for sequence length

        outputs = torch.cat(outputs, dim=1)
        hidden_states = h_t.unsqueeze(0)
        return typing.cast(torch.FloatTensor, outputs), hidden_states


class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(
            self, 
            inputs: torch.FloatTensor, 
            hidden_states: torch.FloatTensor, 
            mask: torch.LongTensor | None = None
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.
        """
        # get the values of the final hidden layer
        h = hidden_states[-1]
        # reshape the hidden state so it can be broadcast to the input shape
        # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        h = h.unsqueeze(1)
        # duplicate the hidden state to match the input shape
        # (batch_size, 1, hidden_size) -> (batch_size, sequence_length, hidden_size)
        h = h.expand_as(inputs)
        # pair hidden states with inputs
        combined = torch.cat([inputs, h], dim=2)
        # project combined to hidden size
        energy = self.W(combined)
        # activation function used in original Bahdanau paper (https://arxiv.org/pdf/1409.0473)
        # - centered at 0 allowing positive and negative values
        # - smooth gradients unlike ReLU
        energy = self.tanh(energy)
        # transform energy so the dot product will yeild meaningful attention scores
        energy = self.V(energy)
        # compute attention scores (scalars)
        x_attention = (energy * inputs).sum(dim=2, keepdim=True)
        # apply mask to attention scores to remove values for padding tokens
        if mask is not None:
            # set masked values to -inf so they will be 0 after softmax
            x_attention = x_attention.masked_fill(mask.unsqueeze(2) == 0, float('-inf'))
        # convert to probabilities
        x_attention = self.softmax(x_attention)
        # apply dropout to prevent overlearning on specific positions
        x_attention = self.dropout(x_attention)
        # elementwise multiply
        outputs = inputs * x_attention
        return outputs, x_attention


class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
            self,
            inputs: torch.FloatTensor,
            hidden_states: torch.FloatTensor
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the token sequences.

        hidden_states
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence.

        hidden_states
            The final hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        # get embeddings for token indices
        x = self.embedding(inputs)
        # apply dropout to regularize the model, should be low for RNN inputs
        # applied after embedding because '0' index in the input corresponds to some token
        # '0' in the embedding corresponds to a meaningful regularization
        x = self.dropout(x)
        # run the bidirectional GRU, which returns the output for each time step and the final hidden state
        # x has shape (batch_size, sequence_length, hidden_size*2)
        # hidden_states has shape (num_layers*2, batch_size, hidden_size)
        x, hidden_states = self.rnn(x, hidden_states)
        # split the forward and backward outputs
        # add them together
        x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]
        # same for hidden states
        final_hidden_states = hidden_states[:self.num_layers] + hidden_states[self.num_layers:]
        return x, typing.cast(torch.FloatTensor, final_hidden_states)

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0


class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        with_attn=True,
    ):

        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        if with_attn:
            self.mlp_attn = Attn(hidden_size, dropout)
        else:
            self.mlp_attn = None

    def forward(
            self,
            inputs: torch.FloatTensor,
            hidden_states: torch.FloatTensor,
            mask: torch.LongTensor | None = None
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Encoder network

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor decoding the orginally encoded input sentence.

        hidden_states
            The final hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        # The decoder must take in the encoder outputs as input and hidden state, and these
        # must be fed into the attention mechanism. The attended input and the encoder hidden state
        # will then be fed into a GRU layer.
        x, hidden_states = self.rnn(inputs, hidden_states)
        if self.mlp_attn is not None:
            x, _ = self.mlp_attn(x, hidden_states, mask)
        return x, hidden_states


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False,
        with_attn=True,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        if not encoder_only:
          self.decoder = DecoderAttn(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            with_attn=with_attn,
          )

    def forward(
            self,
            inputs: torch.LongTensor,
            mask: torch.LongTensor
        ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
         Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor representing the input sentence for sentiment analysis

        hidden_states
            The final hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
            return x[:, 0], hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        return x[:, 0], hidden_states
