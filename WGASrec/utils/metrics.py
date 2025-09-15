import torch
def dcg(r):
    i = torch.arange(1, r.size(-1)+1, device=r.device).float()
    return (r/torch.log2(i+1)).sum(-1)
def ndcg_at_k(scores, target, K):
    b = scores.size(0)
    idx = torch.topk(scores, K, dim=-1).indices
    hit = (idx==target.view(-1,1)).float()
    ideal = torch.ones_like(hit[:, :1])
    return (dcg(hit) / dcg(ideal)).mean().item()
def hit_at_k(scores, target, K):
    idx = torch.topk(scores, K, dim=-1).indices
    hit = (idx==target.view(-1,1)).any(dim=-1).float()
    return hit.mean().item()
