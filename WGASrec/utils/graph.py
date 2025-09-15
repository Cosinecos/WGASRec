import torch
def build_item_cooc(num_items, sequences, window=3, min_weight=1):
    idx = torch.zeros((num_items+1, num_items+1), dtype=torch.float32)
    for seq in sequences:
        s = [int(x) for x in seq if x>0]
        for i in range(len(s)):
            a = s[i]
            j = i+1
            while j<len(s) and j-i<=window:
                b = s[j]
                if a!=b:
                    idx[a,b]+=1
                    idx[b,a]+=1
                j+=1
    idx.fill_diagonal_(0)
    if min_weight>1:
        idx = (idx>=min_weight).float()*idx
    return idx
def normalized_laplacian(A):
    d = A.sum(-1)
    d = torch.where(d>0, d, torch.ones_like(d))
    Dm12 = torch.diag(torch.pow(d, -0.5))
    N = torch.eye(A.size(0), device=A.device) - Dm12@A@Dm12
    return N
def scaled_laplacian(L):
    v = torch.linalg.eigvalsh(L)
    m = float(v.max())
    if m<=0: m=2.0
    return 2.0/m*L - torch.eye(L.size(0), device=L.device)
