import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicImplicitMF(nn.Module):
    """
    Basic Matrix Factorization with implicit feedback.
    Dot product of scientist and paper latent factors,
    plus bias terms for both scientists and papers.
    """
    def __init__(self,
                 num_scientists: int,
                 num_papers: int,
                 embedding_dim: int,
                 global_mean: float,
                 dropout_rate: float = 0.1,
                 implicit_weight: float = 0.2):
        super().__init__()

        self.num_scientists = num_scientists
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.global_mean = global_mean
        self.implicit_weight = implicit_weight

        # Scientist latent factors
        self.scientist_embedding = nn.Embedding(num_scientists, embedding_dim)

        # Paper latent factors
        self.paper_embedding = nn.Embedding(num_papers, embedding_dim)

        # Implicit paper factors (for TBR items)
        self.implicit_embedding = nn.Embedding(num_papers, embedding_dim)

        # Bias terms
        self.scientist_bias = nn.Embedding(num_scientists, 1)
        self.paper_bias = nn.Embedding(num_papers, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.scientist_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.paper_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.implicit_embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scientist_bias.weight)
        nn.init.zeros_(self.paper_bias.weight)

    def forward(self,
                SIDs: torch.Tensor,
                PIDs: torch.Tensor,
                implicit_PIDs: torch.Tensor = None,
                implicit_lengths: torch.Tensor = None
                ) -> torch.Tensor:
        # Get scientist and paper embeddings
        p_s = self.dropout(self.scientist_embedding(SIDs))
        q_p = self.dropout(self.paper_embedding(PIDs))

        # Get bias terms
        b_s = self.scientist_bias(SIDs).squeeze(-1)
        b_p = self.paper_bias(PIDs).squeeze(-1)

        # Process implicit feedback (TBR items) if provided
        if implicit_PIDs is not None and implicit_lengths is not None:
            has_implicit = implicit_lengths > 0

            if torch.any(has_implicit):
                # Create mask for valid items
                mask = torch.arange(implicit_PIDs.size(1), device=SIDs.device)[None, :] < implicit_lengths[:, None]
                mask = mask.unsqueeze(-1).float()

                # Get implicit embeddings and apply mask
                y_j = self.implicit_embedding(implicit_PIDs)
                y_j = y_j * mask

                # Sum the implicit embeddings
                y_sum = y_j.sum(dim=1)
                
                # Normalize by square root of count
                sqrt_lengths = torch.sqrt(implicit_lengths.float() + 1e-8)
                norm_term = (1.0 / sqrt_lengths).unsqueeze(-1)
                y_norm = y_sum * norm_term

                # Enhance user representation with implicit feedback
                p_s = p_s + self.implicit_weight * y_norm

        # Compute interaction term (dot product of latent factors)
        interaction = torch.sum(p_s * q_p, dim=1)

        # Compute final prediction
        prediction = self.global_mean + b_s + b_p + interaction
        return prediction

    def get_l2_reg_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0., device=self.scientist_embedding.weight.device)
        reg_loss += torch.sum(self.scientist_embedding.weight**2)
        reg_loss += torch.sum(self.paper_embedding.weight**2)
        reg_loss += torch.sum(self.implicit_embedding.weight**2)
        reg_loss += 0.5 * torch.sum(self.scientist_bias.weight**2)
        reg_loss += 0.5 * torch.sum(self.paper_bias.weight**2)
        return reg_loss

class AsymmetricSVD(nn.Module):
    """
    Asymmetric SVD that incorporates implicit feedback differently than SVD++

    Instead of learning separate implicit item factors, ASVD uses the same
    item factors for both explicit and implicit interactions.
    """
    def __init__(self,
                 num_scientists: int,
                 num_papers: int,
                 embedding_dim: int,
                 global_mean: float,
                 implicit_weight: float = 0.5,
                 dropout_rate: float = 0.1,
                 sparse_grad: bool = True):
        super().__init__()

        self.num_scientists = num_scientists
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.global_mean = global_mean
        self.implicit_weight = implicit_weight

        # User latent factors
        self.P = nn.Embedding(num_scientists, embedding_dim, sparse=sparse_grad)

        # Item latent factors (used for both explicit and implicit)
        self.Q = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)

        # Bias terms
        self.scientist_bias = nn.Embedding(num_scientists, 1, sparse=sparse_grad)
        self.paper_bias = nn.Embedding(num_papers, 1, sparse=sparse_grad)

        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.P.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Q.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scientist_bias.weight)
        nn.init.zeros_(self.paper_bias.weight)

    def forward(self,
                SIDs: torch.Tensor,
                PIDs: torch.Tensor,
                implicit_PIDs: torch.Tensor,
                implicit_lengths: torch.Tensor) -> torch.Tensor:
        
        # Get explicit embeddings
        p_s = self.dropout(self.P(SIDs))
        q_p = self.dropout(self.Q(PIDs))
        b_s = self.scientist_bias(SIDs).squeeze(-1)
        b_p = self.paper_bias(PIDs).squeeze(-1)

        # Process implicit feedback using the main item embeddings
        mask = torch.arange(implicit_PIDs.size(1), device=implicit_PIDs.device)[None, :] < implicit_lengths[:, None]
        mask = mask.unsqueeze(-1).float()

        # Get implicit item embeddings (using same Q as explicit)
        implicit_q = self.Q(implicit_PIDs)
        implicit_q = implicit_q * mask

        # Sum and normalize
        implicit_sum = implicit_q.sum(dim=1)
        sqrt_lengths = (implicit_lengths.float().sqrt() + 1e-9)
        norm_term = (1.0 / sqrt_lengths).unsqueeze(-1)
        implicit_factors = implicit_sum * norm_term

        # Combine user factors with weighted implicit factors
        user_representation = p_s + self.implicit_weight * implicit_factors

        # Compute final prediction
        interaction = torch.sum(q_p * user_representation, dim=1)
        prediction = self.global_mean + b_s + b_p + interaction

        return prediction

    def get_l2_reg_loss(self):
        reg_loss = torch.tensor(0., device=self.P.weight.device)
        reg_loss += torch.sum(self.P.weight**2)
        reg_loss += torch.sum(self.Q.weight**2)
        reg_loss += torch.sum(self.scientist_bias.weight**2) * 0.5
        reg_loss += torch.sum(self.paper_bias.weight**2) * 0.5
        return reg_loss

class BasicMF(nn.Module):
    """
    This model learns latent factors for scientists and papers along with bias terms
    and predicts ratings as: global_mean + scientist_bias + paper_bias + dot(scientist_factors, paper_factors)
    """
    def __init__(self,
                 num_scientists: int,
                 num_papers: int,
                 embedding_dim: int,
                 global_mean: float,
                 dropout_rate: float = 0.1,
                 sparse_grad: bool = True):
        super().__init__()

        self.num_scientists = num_scientists
        self.num_papers = num_papers
        self.embedding_dim = embedding_dim
        self.global_mean = global_mean

        # Scientist latent factors
        self.P = nn.Embedding(num_scientists, embedding_dim, sparse=sparse_grad)

        # Paper latent factors
        self.Q = nn.Embedding(num_papers, embedding_dim, sparse=sparse_grad)

        # Bias terms
        self.scientist_bias = nn.Embedding(num_scientists, 1, sparse=sparse_grad)
        self.paper_bias = nn.Embedding(num_papers, 1, sparse=sparse_grad)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.P.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.Q.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scientist_bias.weight)
        nn.init.zeros_(self.paper_bias.weight)

    def forward(self,
                SIDs: torch.Tensor,
                PIDs: torch.Tensor,
                *args, **kwargs) -> torch.Tensor:
        p_s = self.dropout(self.P(SIDs))
        q_p = self.dropout(self.Q(PIDs))
        b_s = self.scientist_bias(SIDs).squeeze(-1)
        b_p = self.paper_bias(PIDs).squeeze(-1)
        interaction = torch.sum(p_s * q_p, dim=1)
        prediction = self.global_mean + b_s + b_p + interaction
        return prediction

    def get_l2_reg_loss(self) -> torch.Tensor:
        reg_loss = torch.tensor(0., device=self.P.weight.device)
        reg_loss += torch.sum(self.P.weight**2)
        reg_loss += torch.sum(self.Q.weight**2)
        reg_loss += 0.5 * torch.sum(self.scientist_bias.weight**2)
        reg_loss += 0.5 * torch.sum(self.paper_bias.weight**2)
        return reg_loss

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
        r_hat = global_mean + b_s + b_p + q_p^T * (p_s + |N(s)|^(-1/2) * sum(y_j))
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

class SVDppAG(nn.Module):
    """
    SVD++ with attention and gating
    combines the scientist latent factors using gating with the dot-attention weighted implicit signal

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
        TODO: doc
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
    




class ALS:
    """
    ALS implementation with support for NaN values. 
    Factorizes the ratings matrix A into two rank k matrices U and V such that A = U V^T.

    Parameters
    ----------
    device: torch.device
        The device (CPU or GPU) used for tensor computations.
    train_mat: torch.Tensor
        User-item rating matrix on which the model is trained on.
    """
    def __init__(self, device, train_mat):
        self.device = device
        self.train_mat = train_mat
        self.U = None
        self.V = None

    def predict_with_NaNs(self, train_mat, mean, std):

        # Random initialization of latent factors
        U = torch.randn(train_mat.shape[0], self.rank).to(self.device) * 0.01
        V = torch.randn(train_mat.shape[1], self.rank).to(self.device) * 0.01

        # Alternating optimization
        for i in range(self.iterations):
            U = self.optimize_U_with_NaNs(U, V, train_mat)
            V = self.optimize_V_with_NaNs(U, V, train_mat)

        self.U = U
        self.V = V
    
    def optimize_U_with_NaNs(self, U, V, A):
        assert U.shape[0] == A.shape[0] and U.shape[1] == V.shape[1] and V.shape[0] == A.shape[1]

        # Optimize matrix U while keeping V fixed
        n = U.shape[0]
        m = V.shape[0]
        rank = U.shape[1]

        A_masked = torch.nan_to_num(A, nan=0.0)
        B = V.T @ A_masked.T
        Id_lam = self.lam * torch.eye(rank, dtype=U.dtype, device=U.device)

        # Precompute of V_i^T V_i
        Q = V.unsqueeze(2) * V.unsqueeze(1)

        for j in range(n):
            
            mat = Id_lam.clone()

            valid_indices = (~torch.isnan(A[j, :])).nonzero(as_tuple=True)[0]

            if valid_indices.numel() > 0:
                mat += Q[valid_indices].sum(dim=0)

            U[j] = torch.linalg.solve(mat, B[:, j])

        return U
    
    def optimize_V_with_NaNs(self, U, V, A):
        assert U.shape[0] == A.shape[0] and U.shape[1] == V.shape[1] and V.shape[0] == A.shape[1]

        # Optimize matrix V while keeping U fixed
        n = U.shape[0]
        m = V.shape[0]
        rank = U.shape[1]

        A_masked = torch.nan_to_num(A, nan=0.0)
        B = U.T @ A_masked
        Id_lam = self.lam * torch.eye(rank, dtype=U.dtype, device=U.device)

        # Precompute of U_i^T U_i
        Q = U.unsqueeze(2) * U.unsqueeze(1)

        for j in range(m):
            mat = Id_lam.clone()

            valid_indices = (~torch.isnan(A[:, j])).nonzero(as_tuple=True)[0]
            if valid_indices.numel() > 0:
                mat += Q[valid_indices].sum(dim=0)

            V[j] = torch.linalg.solve(mat, B[:, j])

        return V
    
    def train(self, lam, rank, iterations):
        """
        Train the ALS model.

        Parameters
        ----------
        lam: float
            Regularization factor
        rank: int
            Rank of matrices U and V
        iterations: int
            Number of optimization iterations
        """
        self.lam = lam
        self.rank = rank
        self.iterations = iterations

        self.predict_with_NaNs(self.train_mat, None, None)

    def get_predictions_matrix(self):

        # Return the predicted ratings as a matrix
        return self.U @ self.V.T
    
