import torch

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


# def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
#     """ Perform Differentiable Optimal Transport in Log-space for stability"""
#     b, m, n = scores.shape
#     one = scores.new_tensor(1)
#     ms, ns = (m*one).to(scores), (n*one).to(scores)

#     bins0 = alpha.expand(b, m, 1)
#     bins1 = alpha.expand(b, 1, n)
#     alpha = alpha.expand(b, 1, 1)

#     couplings = torch.cat([torch.cat([scores, bins0], -1),
#                            torch.cat([bins1, alpha], -1)], 1)

#     norm = - (ms + ns).log()
#     log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
#     log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
#     log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

#     print('couplings:', couplings.shape)
#     print('log_mu:', log_mu.shape)
#     print('log_nu:', log_nu.shape)
#     Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
#     Z = Z - norm  # multiply probabilities by M+N
#     return Z

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    norm = - (ms + ns).log()
    log_mu = norm.expand(m)
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)

    Z = Z - norm  # multiply probabilities by M+N
    return Z


def solve_optimal_transport(scores, iters, match_threshold):
    # Run the optimal transport.
    scores = log_optimal_transport(
        scores, info.bin_score,
        iters=iters)

    # Get the matches with score above "match_threshold".
    max0, max1 = scores[:, :, :].max(2), scores[:, :, :].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return {
        'matches0': indices0, # use -1 for invalid match
        'matches1': indices1, # use -1 for invalid match
        'matching_scores0': mscores0,
        'matching_scores1': mscores1,
    }
    
if __name__ == '__main__':
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }
    class Info:
        def __init__(self):
            pass
    info = Info()
    info.config = default_config
    info.bin_score = torch.tensor(1)
 
    score = torch.rand(1, 2, 4)
    solve_optimal_transport(info, score)
