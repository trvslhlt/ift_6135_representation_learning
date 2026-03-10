import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# original paper: https://arxiv.org/pdf/1607.06450
class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        # eps is primarily used for numerical stability
        self.eps = eps
        # from the paper, weight is 'g' (gain)
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        # and bias is 'b' (bias)
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        # initialize to unit scale and 0 bias
        self.reset_parameters()

    def forward(self, inputs: torch.FloatTensor) -> torch.FloatTensor:
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """
        # compute the mean across the last dimension (feature values)
        # keep the input dimensions for broadcasting
        mean = inputs.mean(dim=-1, keepdim=True)
        # same for variance
        # use the biased variance per the paper (eqs 3, 4)
        # why? "In LayerNorm, we aren't trying to "estimate" a hidden population parameter; 
        # we are simply trying to re-scale a vector so it is easier to optimize"
        var = inputs.var(dim=-1, keepdim=True, unbiased=False)
        # normalize the inputs
        x = (inputs - mean) / torch.sqrt(var + self.eps)
        # rescale and shift using the learnable parameters ('g' and 'b')
        output = self.weight * x + self.bias
        return typing.cast(torch.FloatTensor, output)
        

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size: int, num_heads: int):
        super().__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        embed_dim = head_size * num_heads
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_y = nn.Linear(embed_dim, embed_dim)

        # ==========================
        # TODO: Write your code here
        # ==========================

    def get_attention_weights(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            mask: torch.LongTensor | None = None
        ) -> torch.FloatTensor:
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """
        # queries: `(batch_size, num_heads, sequence_length, head_size)`
        # keys: `(batch_size, num_heads, sequence_length, head_size)`
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_size)
        # sequences may have different lengths but the tensors need to be rectangular
        # so positions of shorter sequences are paddeed
        # we apply masking so attention ignores padded positions
        if mask is not None:
            # mask: `(batch_size, sequence_length)`
            # unsqueeze so it has shape `(batch_size, 1, 1, sequence_length)`
            mask1 = mask.unsqueeze(1).unsqueeze(2)
            # set the masked positions to a very large negative value (e.g. -1e9)
            scores = scores.masked_fill(mask1 == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return typing.cast(torch.FloatTensor, weights)

    def apply_attention(
            self,
            queries: torch.FloatTensor,
            keys: torch.FloatTensor,
            values: torch.FloatTensor,
            mask: torch.LongTensor | None = None
        ) -> torch.FloatTensor:
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads.

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence.
        """
        # get attention weights
        weights = self.get_attention_weights(queries, keys, mask)
        # compute the attended values
        attended_values = weights @ values
        # merge heads
        outputs = self.merge_heads(typing.cast(torch.FloatTensor, attended_values))
        return typing.cast(torch.FloatTensor, outputs)

    def split_heads(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        # tensor: `(batch_size, sequence_length, num_heads * dim)`
        batch_size, sequence_length, _ = tensor.shape
        # create a new dimension for the heads
        # t2: `(batch_size, sequence_length, num_heads, dim)`
        t2 = tensor.reshape(batch_size, sequence_length, self.num_heads, -1)
        # output: `(batch_size, num_heads, sequence_length, dim)`
        output = t2.transpose(1, 2)
        return typing.cast(torch.FloatTensor, output)

    def merge_heads(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """
        # tensor: `(batch_size, num_heads, sequence_length, dim)`
        batch_size, _, sequence_length, dim = tensor.shape
        # t2: `(batch_size, sequence_length, num_heads, dim)`
        t2 = tensor.transpose(1, 2)
        # consolidate the heads into a single dimension
        # use `self.num_heads` over the value read from the tensor to catch inconsistencies
        output = t2.reshape(batch_size, sequence_length, self.num_heads * dim)
        return typing.cast(torch.FloatTensor, output)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            mask: torch.LongTensor | None = None
        ) -> torch.FloatTensor:
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """
        # compute heads for queries, keys, and values with linear projections
        # all heads: `(batch_size, sequence_length, num_heads * head_size)`
        q_heads = self.w_q(hidden_states)
        k_heads = self.w_k(hidden_states)
        v_heads = self.w_v(hidden_states)
        # split the heads into a separate dimension
        # queries, keys, and values: `(batch_size, num_heads, sequence_length, head_size)`
        q = self.split_heads(q_heads)
        k = self.split_heads(k_heads)
        v = self.split_heads(v_heads)
        # compute the attended values and merge the heads
        # y: `(batch_size, sequence_length, num_heads * head_size)`
        y = self.apply_attention(q, k, v, mask)
        # compute output with linear projection
        output = self.w_y(y)
        return typing.cast(torch.FloatTensor, output)


class PostNormAttentionBlock(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_heads: int, dropout: float = 0.30):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(head_size=embed_dim//num_heads, num_heads=num_heads)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(
        self, x: torch.FloatTensor, 
        mask: torch.LongTensor | None = None
    ) -> torch.FloatTensor:
        attention_outputs = self.attn(x, mask)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs + attention_outputs)
        return outputs

class PreNormAttentionBlock(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.0):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.

        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(head_size=embed_dim//num_heads, num_heads=num_heads)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, mask=None):
        # ==========================
        # TODO: Write your code here
        # ==========================
        pass



class Transformer(nn.Module):

    def __init__(
        self,
        vocabulary_size: int = 30522,
        sequence_length: int = 256,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 1,
        num_layers: int = 2,
        block: str = "prenorm",
        dropout=0.3,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            block - Type of attention block
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        #Adding the cls token to the sequnence
        self.sequence_length= 1 + sequence_length
        # Layers/Networks
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        if block =="prenorm":
          self.transformer = nn.ModuleList([PreNormAttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        elif block =="postnorm":
          self.transformer = nn.ModuleList([PostNormAttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        else:
          raise ValueError(f"Invalid block type {block}")
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.sequence_length,embed_dim))

    def forward(
            self,
            x: torch.LongTensor,
            mask: torch.LongTensor | None = None
        ) -> torch.FloatTensor:
        """Transformer

        This is a small version of  Transformer

        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, sequence length)`)
            The input tensor containing text.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input

        x1 = self.embedding(x)
        B, T, _ = x.shape

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x2 = torch.cat([cls_token, x1], dim=1)
        x3 = x2 + self.pos_embedding[:,:T+1]
        # Add dropout and then the transformer (remember to update the mask because of the CLS token)
        # ==========================
        # TODO: Write your code here
        # ==========================


        # Take the cls token representation and send it to mlp_head
        # ==========================
        # TODO: Write your code here
        # ==========================
        pass
