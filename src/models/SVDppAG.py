import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SVDppAG(nn.Module):
    """
    SVD++ with attention and gating
    combines the scientist latent factors using gating with the dot-attention weighted implicit signal

    If `use_attention` and `use_gating` are both set to `False`, this model behaves like the original SVD++.

    Parameters
    ----------
    num_scientists: int
        Number of scientists.
    num_papers: int
        Number of papers.
    embedding_dim: int
        Dimensionality of the embedding space.
    global_mean: float
        Global mean rating.
    sparse_grad: bool, default=True
        use sparse gradients for embedding layers.
    use_attention: bool, default=True
        use dot-attention to combine implicit signal, else use scaled sum (as SVD++).
    use_gating: bool, default=True
        use gating to combine scientist factor and transformed implicit factor, else use sum (as SVD++).

    Returns
    -------
    torch.Tensor
        Predicted ratings (shape: [batch_size]).
    """
    def __init__(self, 
                 num_scientists: int, 
                 num_papers: int, 
                 embedding_dim: int, 
                 global_mean: float,
                 sparse_grad: bool = True,
                 use_attention: bool = True,
                 use_gating: bool = True,
        ):
        super().__init__()

        self.num_scientists = num_scientists
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.global_mean = global_mean
        self.use_attention = use_attention
        self.use_gating = use_gating

        # Scientist factors
        self.P = nn.Embedding(num_scientists, embedding_dim, sparse=sparse_grad)

        # Paper factor
        self.Q = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)

        # Implicit paper factors
        self.Y = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)

        # Scientist bias
        self.Bs = nn.Embedding(num_scientists, 1, sparse=sparse_grad)

        # Paper bias
        self.Bp = nn.Embedding(num_papers, 1, sparse=sparse_grad)

                
        if use_gating:
            # Gating to combine scientist factor and transformed implicit factor
            self.gate = nn.Linear(2 * embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initializes embedding weights."""
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.normal_(self.Y.weight, std=0.01)
        nn.init.zeros_(self.Bs.weight)
        nn.init.zeros_(self.Bp.weight)

        if self.use_gating:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, 
                SIDs: torch.Tensor,
                PIDs: torch.Tensor,
                implicit_PIDs: torch.Tensor,
                implicit_lengths: torch.Tensor
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        SIDs: torch.Tensor
            Tensor of scientist IDs (shape: [batch_size]).
        PIDs: torch.Tensor
            Tensor of paper IDs (shape: [batch_size]).
        implicit_PIDs: torch.Tensor
            Tensor of implicit paper IDs (shape: [batch_size, max_implicit_length]).
        implicit_lengths: torch.Tensor
            Tensor of lengths of implicit signals (shape: [batch_size]).
        """
        p_s = self.P(SIDs)
        q_p = self.Q(PIDs)
        b_s = self.Bs(SIDs).squeeze()
        b_p = self.Bp(PIDs).squeeze()
        y_j = self.Y(implicit_PIDs)

        # Mask implicit signal
        max_len = implicit_PIDs.size(1)
        idx_arange = torch.arange(max_len, device=implicit_PIDs.device)
        implicit_mask = idx_arange[None, :] < implicit_lengths[:, None]

        processed_implicit_signal: torch.Tensor
        
        if self.use_attention:
            query = p_s.unsqueeze(1)
            attn_scores = torch.bmm(query, y_j.transpose(1, 2)) / math.sqrt(self.embedding_dim)
            attn_scores.masked_fill_(~implicit_mask.unsqueeze(1), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attended_y = torch.bmm(attn_weights, y_j)
            processed_implicit_signal = attended_y.squeeze(1)
        else:
            # Original SVD++ style aggregation
            y_j_masked = y_j * implicit_mask.unsqueeze(-1).float()
            y_sum = y_j_masked.sum(dim=1) # (batch_size, embedding_dim)

            # Norm_term = |N(s)|^(-1/2)
            sqrt_lengths = (implicit_lengths.float().sqrt() + 1e-9)
            norm_term = (1.0 / sqrt_lengths).unsqueeze(-1)
            processed_implicit_signal = y_sum * norm_term

        # Gating to combine scientist factor and transformed implicit factor
        if self.use_gating:
            gate_input = torch.cat([p_s, processed_implicit_signal], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            user_representation = gate * p_s + (1 - gate) * processed_implicit_signal
        else:
            # Simple sum if not using gating
            user_representation = p_s + processed_implicit_signal

        # Final interaction
        interaction = torch.sum(q_p * user_representation, dim=1)
        prediction = self.global_mean + b_s + b_p + interaction

        return prediction

    def get_l2_reg_loss(self) -> torch.Tensor:
        """
        get L2 regularization loss for all embeddings.
        """
        reg_loss = torch.tensor(0., device=self.P.weight.device)
        reg_loss += torch.sum(self.P.weight**2)
        reg_loss += torch.sum(self.Q.weight**2)
        reg_loss += torch.sum(self.Y.weight**2)
        reg_loss += torch.sum(self.Bs.weight**2)
        reg_loss += torch.sum(self.Bp.weight**2)

        # Gate parameters
        if self.use_gating:
            reg_loss += torch.sum(self.gate.weight**2)
            if self.gate.bias is not None:
                reg_loss += torch.sum(self.gate.bias**2)
        return reg_loss
