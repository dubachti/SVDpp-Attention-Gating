import torch
import torch.nn as nn

class SVDpp(nn.Module):
    """
    SVD++ 

    Parameters
    ----------
    num_scientists : int
        Number of scientists.
    num_papers : int
        Number of papers.
    embedding_dim : int
        Dimensionality of the embedding.
    global_mean : float
        Global mean rating.
    sparse_grad : bool, default=True
        Use sparse gradients for embedding layers.
    """
    def __init__(self, 
                 num_scientists: int, 
                 num_papers: int, 
                 embedding_dim: int, 
                 global_mean: float,
                 sparse_grad: bool = True):
        super().__init__()

        self.num_scientists = num_scientists
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.global_mean = global_mean

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


        self._init_weights()

    def _init_weights(self):
        """Initializes embedding weights."""
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.normal_(self.Y.weight, std=0.01)
        nn.init.zeros_(self.Bs.weight)
        nn.init.zeros_(self.Bp.weight)

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

        Returns
        -------
        torch.Tensor
            Predicted ratings (shape: [batch_size]).
        """
        p_s = self.P(SIDs)
        q_p = self.Q(PIDs)
        b_s = self.Bs(SIDs).squeeze()
        b_p = self.Bp(PIDs).squeeze()
        y_j = self.Y(implicit_PIDs)

        # Mask implicit signal
        mask = torch.arange(implicit_PIDs.size(1), device=implicit_PIDs.device)[None, :] < implicit_lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        y_j = y_j * mask
        y_sum = y_j.sum(dim=1)

        # Norm_term = |N(s)|^(-1/2)
        sqrt_lengths =  (implicit_lengths.sqrt() + 1e-9)
        norm_term = (1.0 / sqrt_lengths).unsqueeze(-1)
        y_norm = y_sum * norm_term

        # Interaction = q_p^T * (p_s + |N(s)|^(-1/2) * sum(y_j))
        interaction = torch.sum(q_p * (p_s + y_norm), dim=1)

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
        return reg_loss 