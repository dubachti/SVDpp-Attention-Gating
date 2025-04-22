
## SVD++ estimate:
$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T \left( p_u + |N(u)|^{-\frac{1}{2}} \sum_{j \in N(u)} y_j \right)$$

## SVD++ MLP estimate:
$$\hat{r}_{ui} = \mu + b_u + b_i + q_i^T \left( \mathbf{gate} \left( p_u, \mathbf{MLP} \left(|N(u)|^{-\frac{1}{2}} \sum_{j \in N(u)} y_j \right) \right) \right)$$

where 
- $\mu$ is the global average rating
- $b_u$ is the bias of user $u$
- $b_i$ is the bias of item $i$
- $q_i$ is the latent factor of item $i$
- $p_u$ is the latent factor of user $u$
- $N(u)$ is the set of items rated by user $u$
- $y_j$ is the latent factor of implicit item $j$
- $\mathbf{gate}$ is something like ~ gate(x,y) = $\alpha * x + (1 - \alpha) * y$
