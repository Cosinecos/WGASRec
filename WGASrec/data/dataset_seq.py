import os, json, random, torch
from torch.utils.data import Dataset
from ..utils.registry import register
PAD=0
def sym_extend(seq, need):
    if not seq: return [PAD]*need
    body = seq[1:-1] if len(seq)>2 else []
    mirror = list(reversed(body))
    ext = [seq[0]]+mirror+[seq[-1]]
    out=[];i=0
    while len(out)<need:
        out.append(ext[i%len(ext)]);i+=1
    return out
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    pad_len = max_len - len(seq)
    if pad_len>0:
        ext = sym_extend(seq, pad_len)
        if len(ext)<pad_len:
            ext = [PAD]*(pad_len-len(ext))+ext
        seq = ext+seq
    return seq, pad_len
def attn_mask(pad_len, max_len):
    return [0.0]*pad_len + [1.0]*(max_len-pad_len)
def positions(pad_len, max_len):
    return [0]*pad_len + list(range(1, max_len-pad_len+1))
def sample_neg(num_items, forbid):
    while True:
        x = random.randint(1, num_items)
        if x not in forbid: return x
class SeqDataset(Dataset):
    def __init__(self, jsonl_path, max_len, num_items=None, enable_bpr_neg=False, boundary_tokens_per_end=0):
        self.max_len=max_len
        self.samples=[]
        with open(jsonl_path,'r',encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.num_items=num_items
        self.enable_bpr_neg=enable_bpr_neg
        self.boundary_tokens_per_end=int(boundary_tokens_per_end)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x=self.samples[idx]
        u=int(x.get('user',0))
        seq=list(map(int,x['seq']))
        tgt=int(x['target'])
        item={
            'user':torch.tensor(u,dtype=torch.long),
            'seq':torch.tensor(seq_pad,dtype=torch.long),
            'mask':torch.tensor(attn_mask(n_pad,self.max_len),dtype=torch.float32),
            'positions':torch.tensor(positions(n_pad,self.max_len),dtype=torch.long),
            'target':torch.tensor(tgt,dtype=torch.long),
            'n_pad':torch.tensor(n_pad,dtype=torch.long),
            'tau_per_end':torch.tensor(self.boundary_tokens_per_end,dtype=torch.long),
            'seq_len':torch.tensor(min(len(seq),self.max_len),dtype=torch.long)
        }
        if self.enable_bpr_neg and self.num_items:
            forbid=set(seq)|{tgt,0}
            item['neg']=torch.tensor(sample_neg(self.num_items,forbid),dtype=torch.long)
        return item
def collate_fn(batch):
    out={}
    for k in batch[0].keys():
        if isinstance(batch[0][k],torch.Tensor):
            out[k]=torch.stack([b[k] for b in batch],0)
        else:
            out[k]=[b[k] for b in batch]
    return out
def _p(d,n): return os.path.join(d,f'{n}.jsonl')
def build_generic(data_dir,max_len,enable_bpr_neg=False,boundary_tokens_per_end=0):
    meta=json.load(open(os.path.join(data_dir,'meta.json'),'r',encoding='utf-8'))
    n=meta.get('num_items')
    tr=SeqDataset(_p(data_dir,'train'),max_len,n,enable_bpr_neg,boundary_tokens_per_end)
    te=SeqDataset(_p(data_dir,'test'),max_len,n,False,boundary_tokens_per_end)
    return tr,va,te,meta
@register('dataset','ml1m')
def build_ml1m(data_dir,max_len,enable_bpr_neg=False,boundary_tokens_per_end=0):
    return build_generic(data_dir,max_len,enable_bpr_neg,boundary_tokens_per_end)
@register('dataset','amazon_beauty')
def build_beauty(data_dir,max_len,enable_bpr_neg=False,boundary_tokens_per_end=0):
    return build_generic(data_dir,max_len,enable_bpr_neg,boundary_tokens_per_end)
@register('dataset','amazon_sports')
def build_sports(data_dir,max_len,enable_bpr_neg=False,boundary_tokens_per_end=0):
    return build_generic(data_dir,max_len,enable_bpr_neg,boundary_tokens_per_end)
@register('dataset','lastfm')
def build_lastfm(data_dir,max_len,enable_bpr_neg=False,boundary_tokens_per_end=0):
    return build_generic(data_dir,max_len,enable_bpr_neg,boundary_tokens_per_end)
