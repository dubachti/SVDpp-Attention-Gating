import torch
import torch.nn as nn

class EmbeddingDotProductModel(nn.Module):
    def __init__(self, n_sids: int, n_pids: int, embedding_dim: int):
        super().__init__()
        # embedding layers for scientists and papers
        self.scientist_embedding = nn.Embedding(n_sids, embedding_dim)
        self.paper_embedding = nn.Embedding(n_pids, embedding_dim)
        # initialize weights
        nn.init.xavier_uniform_(self.scientist_embedding.weight)
        nn.init.xavier_uniform_(self.paper_embedding.weight)

    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        scientist_vec = self.scientist_embedding(sid) # (batch_size, embedding_dim)
        paper_vec = self.paper_embedding(pid)       # (batch_size, embedding_dim)
        return torch.sum(scientist_vec * paper_vec, dim=-1) # (batch_size)

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

        # scientist factors
        self.P = nn.Embedding(num_scientists, embedding_dim, sparse=sparse_grad)
        # paper factor
        self.Q = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)
        # implicit paper factors
        self.Y = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)
        # scientist bias
        self.Bs = nn.Embedding(num_scientists, 1, sparse=sparse_grad)
        # paper bias
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
                SIDs: torch.Tensor, # (batch_size,)
                PIDs: torch.Tensor, # (batch_size,)
                implicit_PIDs: torch.Tensor, # (batch_size, max_implicit_len)
                implicit_lengths: torch.Tensor # (batch_size,)
                ) -> torch.Tensor:
        """
        r_hat = global_mean + b_s + b_p + q_p^T * (p_s + |N(s)|^(-1/2) * sum(y_j))
        """
        p_s = self.P(SIDs) # (batch_size, embedding_dim)
        q_p = self.Q(PIDs) # (batch_size, embedding_dim)
        b_s = self.Bs(SIDs).squeeze() # (batch_size,)
        b_p = self.Bp(PIDs).squeeze() # (batch_size,)
        y_j = self.Y(implicit_PIDs) # (batch_size, max_implicit_len, embedding_dim)

        # # y_sum = sum(y_j)
        ##########
        mask = torch.arange(implicit_PIDs.size(1), device=implicit_PIDs.device)[None, :] < implicit_lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        y_j = y_j * mask
        y_sum = y_j.sum(dim=1)
        ######## # tf does below code not work instead of above
        # y_sum = y_j.sum(dim=1) # (batch_size, embedding_dim)

        # norm_term = |N(s)|^(-1/2)
        sqrt_lengths =  (implicit_lengths.sqrt() + 1e-9) # (batch_size,)
        norm_term = (1.0 / sqrt_lengths).unsqueeze(-1) # (batch_size, 1)
        y_norm = y_sum * norm_term # (batch_size, embedding_dim)

        # interaction = q_p^T * (p_s + |N(s)|^(-1/2) * sum(y_j))
        interaction = torch.sum(q_p * (p_s + y_norm), dim=1) # (batch_size,)

        prediction = self.global_mean + b_s + b_p + interaction # (batch_size,)

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
    
class SVDppMLP(nn.Module):
    """
    SVD++ MLP 

    combines the scientist latent factors with the MLP transformed implicit signal

    Parameters
    ----------
    num_scientists: int
        Number of scientists.
    num_papers: int
        Number of papers.
    embedding_dim: int
        Dimensionality of the embedding.
    global_mean: float
        Global mean rating.
    use_gating: bool, default=True
        if True, use gating to combine scientist factor and transformed implicit factor, else use simple sum.

    """
    def __init__(self, 
                 num_scientists: int, 
                 num_papers: int, 
                 embedding_dim: int, 
                 global_mean: float,
                 use_gating: bool = True,
                 MLP_layers: int = 1,
                 sparse_grad: bool = True,
        ):
        super().__init__()

        self.num_scientists = num_scientists
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.global_mean = global_mean
        self.use_gating = use_gating

        # scientist factors
        self.P = nn.Embedding(num_scientists, embedding_dim, sparse=sparse_grad)
        # paper factor
        self.Q = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)
        # implicit paper factors
        self.Y = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)
        # scientist bias
        self.Bs = nn.Embedding(num_scientists, 1, sparse=sparse_grad)
        # paper bias
        self.Bp = nn.Embedding(num_papers, 1, sparse=sparse_grad)
        # MLP for implicit signal
        self.implicit_MLP = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
            )
        for _ in range(MLP_layers - 1):
            self.implicit_MLP.append(nn.ReLU())
            self.implicit_MLP.append(nn.Linear(embedding_dim, embedding_dim))
            
        if use_gating:
            # gating to combine scientist factor and transformed implicit factor
            self.gate = nn.Linear(2 * embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Initializes embedding weights."""
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.normal_(self.Y.weight, std=0.01)
        nn.init.zeros_(self.Bs.weight)
        nn.init.zeros_(self.Bp.weight)
        for param in self.implicit_MLP.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif isinstance(param, nn.Parameter) and param.requires_grad:
                nn.init.zeros_(param)

        if self.use_gating:
            nn.init.xavier_uniform_(self.gate.weight)
            if self.gate.bias is not None:
                nn.init.zeros_(self.gate.bias)

    def forward(self, 
                SIDs: torch.Tensor, # (batch_size,)
                PIDs: torch.Tensor, # (batch_size,)
                implicit_PIDs: torch.Tensor, # (batch_size, max_implicit_len)
                implicit_lengths: torch.Tensor # (batch_size,)
                ) -> torch.Tensor:
        """
        r_hat = global_mean + b_s + b_p + q_p^T * (p_s + |N(s)|^(-1/2) * sum(y_j))
        """
        p_s = self.P(SIDs) # (batch_size, embedding_dim)
        q_p = self.Q(PIDs) # (batch_size, embedding_dim)
        b_s = self.Bs(SIDs).squeeze() # (batch_size,)
        b_p = self.Bp(PIDs).squeeze() # (batch_size,)
        y_j = self.Y(implicit_PIDs) # (batch_size, max_implicit_len, embedding_dim)

        # # y_sum = sum(y_j)
        ##########
        mask = torch.arange(implicit_PIDs.size(1), device=implicit_PIDs.device)[None, :] < implicit_lengths[:, None]
        mask = mask.unsqueeze(-1).float()
        y_j = y_j * mask
        y_sum = y_j.sum(dim=1)
        ######## # tf does below code not work instead of above
        # y_sum = y_j.sum(dim=1) # (batch_size, embedding_dim)

        # norm_term = |N(s)|^(-1/2)
        sqrt_lengths =  (implicit_lengths.sqrt() + 1e-9) # (batch_size,)
        norm_term = (1.0 / sqrt_lengths).unsqueeze(-1) # (batch_size, 1)
        y_norm = y_sum * norm_term # (batch_size, embedding_dim)

        # project implicit signal to the same space as scientist
        transformed_y = self.implicit_MLP(y_norm) # (batch_size, embedding_dim)
        # gating to combine scientist factor and transformed implicit factor
        if self.use_gating:
            gate_input = torch.cat([p_s, transformed_y], dim=-1) # (batch_size, 2*embedding_dim)
            gate = torch.sigmoid(self.gate(gate_input)) # (batch_size, 1)
            user_representation = gate * p_s + (1 - gate) * transformed_y # (batch_size, embedding_dim)
        else:
            user_representation = p_s + transformed_y

        # interaction = q_p^T * (gate(p_s, MLP(|N(s)|^(-1/2) * sum(y_j))))
        interaction = torch.sum(q_p * user_representation, dim=1) # (batch_size,)
        prediction = self.global_mean + b_s + b_p + interaction # (batch_size,)

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
        # MLP parameters
        for param in self.implicit_MLP.parameters():
            if param.requires_grad:
                reg_loss += torch.sum(param**2)
        # Gate parameters
        if self.use_gating:
            reg_loss += torch.sum(self.gate.weight**2)
            if self.gate.bias is not None:
                reg_loss += torch.sum(self.gate.bias**2)
        return reg_loss
