import os, yaml, torch, argparse, numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils.seed import set_seed
from .utils.registry import get
from .data.dataset_seq import collate_fn
from .utils.graph import build_item_cooc, normalized_laplacian, scaled_laplacian
from .utils.metrics import hit_at_k, ndcg_at_k
def load_cfg(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)
def build_graph(num_items, train_jsonl):
    seqs=[]
    with open(os.path.join(train_jsonl),'r',encoding='utf-8') as f:
        for line in f:
            x=eval(line) if line.strip().startswith('{') else None
    with open(os.path.join(os.path.dirname(train_jsonl),'train.jsonl'),'r',encoding='utf-8') as f:
        for line in f:
            x=eval(line)
            seqs.append(x['seq'])
    A=build_item_cooc(num_items, seqs, window=3, min_weight=1)
    L=normalized_laplacian(A)
    Lt=scaled_laplacian(L)
    return Lt
def make_loaders(ds, bs, nw):
    tr,va,te,_=ds
    ltr=DataLoader(tr,batch_size=bs,shuffle=True,num_workers=nw,collate_fn=collate_fn,pin_memory=True)
    lva=DataLoader(va,batch_size=bs,shuffle=False,num_workers=nw,collate_fn=collate_fn,pin_memory=True)
    lte=DataLoader(te,batch_size=bs,shuffle=False,num_workers=nw,collate_fn=collate_fn,pin_memory=True)
    return ltr,lva,lte
def sample_100(scores, target, num_items):
    b=scores.size(0)
    dev=scores.device
    neg=torch.randint(1,num_items+1,(b,100),device=dev)
    neg[neg==target.view(-1,1)]=1
    cand=torch.cat([target.view(-1,1),neg],1)
    s=torch.gather(scores,1,cand)
    return s
def evaluate(model, loader, num_items, device):
    model.eval()
    H10=0; H20=0; N10=0; N20=0; n=0
    with torch.no_grad():
        for batch in loader:
            seq=batch['seq'].to(device)
            mask=batch['mask'].to(device)
            tau=batch['tau_per_end'].to(device)
            tgt=batch['target'].to(device)
            logits=model(seq,mask,tau)
            s=sample_100(logits,tgt,num_items)
            H10+=hit_at_k(s, torch.zeros(seq.size(0),device=device, dtype=torch.long), 10)*seq.size(0)
            H20+=hit_at_k(s, torch.zeros(seq.size(0),device=device, dtype=torch.long), 20)*seq.size(0)
            N10+=ndcg_at_k(s, torch.zeros(seq.size(0),device=device, dtype=torch.long), 10)*seq.size(0)
            N20+=ndcg_at_k(s, torch.zeros(seq.size(0),device=device, dtype=torch.long), 20)*seq.size(0)
            n+=seq.size(0)
    return H10/n, N10/n, H20/n, N20/n
def train_one(cfg):
    set_seed(cfg.get('seed',42))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=get('dataset',cfg['dataset']['name'])
    tr,va,te,meta=data(cfg['data_dir'], cfg['dataset']['max_len'], enable_bpr_neg=True, boundary_tokens_per_end=cfg['model'].get('boundary_tokens_per_end',0))
    ltr,lva,lte=make_loaders((tr,va,te,meta), cfg['train']['batch_size'], cfg['train'].get('num_workers',0))
    Lt=build_item_cooc(meta['num_items'], [])
    L=normalized_laplacian(Lt)
    Ls=scaled_laplacian(L)
    Ls=Ls.to(device)
    model=get('model','WGASRec')(num_items=meta['num_items'], hidden_size=cfg['model']['hidden_size'], max_seq_length=cfg['dataset']['max_len'], num_layers=cfg['model']['num_layers'], dropout=cfg['model']['dropout'], J=cfg['model']['decomp_level'], gate_tau=cfg['model'].get('gate_tau',1.0), Lg=cfg['model']['Lg'], K=cfg['model']['K'], Ltilde=Ls)
    model=model.to(device)
    opt=torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    best=-1; patience=cfg['train']['patience']; out=cfg['out']
    os.makedirs(out, exist_ok=True)
    for epoch in range(cfg['train']['epochs']):
        model.train()
        pbar=tqdm(ltr, ncols=100)
        for batch in pbar:
            seq=batch['seq'].to(device)
            mask=batch['mask'].to(device)
            tau=batch['tau_per_end'].to(device)
            tgt=batch['target'].to(device)
            logits=model(seq,mask,tau)
            loss=F.cross_entropy(logits, tgt, ignore_index=0)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_description(f'ep{epoch} loss {loss.item():.4f}')
        h10,n10,h20,n20=evaluate(model,lva,meta['num_items'],device)
        score=n10
        if score>best:
            best=score
            torch.save({'model':model.state_dict(),'cfg':cfg}, os.path.join(out,'best.pt'))
            wait=0
        else:
            wait+=1
            if wait>=patience: break
    return os.path.join(out,'best.pt')
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=str,required=True)
    ap.add_argument('--data_dir',type=str,required=True)
    ap.add_argument('--out',type=str,required=True)
    args=ap.parse_args()
    cfg=load_cfg(args.config)
    cfg['data_dir']=args.data_dir
    cfg['out']=args.out
    train_one(cfg)
if __name__=='__main__':
    main()
