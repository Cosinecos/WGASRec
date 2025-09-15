import os, yaml, torch, argparse
from torch.utils.data import DataLoader
from .utils.registry import get
from .data.dataset_seq import collate_fn
from .train import evaluate
def load_cfg(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=str,required=True)
    ap.add_argument('--data_dir',type=str,required=True)
    ap.add_argument('--ckpt',type=str,required=True)
    args=ap.parse_args()
    cfg=load_cfg(args.config)
    data=get('dataset',cfg['dataset']['name'])
    tr,va,te,meta=data(args.data_dir, cfg['dataset']['max_len'], enable_bpr_neg=False, boundary_tokens_per_end=cfg['model'].get('boundary_tokens_per_end',0))
    lte=DataLoader(te,batch_size=cfg['train']['batch_size'],shuffle=False,collate_fn=collate_fn)
    ck=torch.load(args.ckpt,map_location='cpu')
    model=get('model','WGASRec')(num_items=meta['num_items'], hidden_size=cfg['model']['hidden_size'], max_seq_length=cfg['dataset']['max_len'], num_layers=cfg['model']['num_layers'], dropout=cfg['model']['dropout'], J=cfg['model']['decomp_level'], gate_tau=cfg['model'].get('gate_tau',1.0), Lg=cfg['model']['Lg'], K=cfg['model']['K'], Ltilde=None)
    model.load_state_dict(ck['model'], strict=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    h10,n10,h20,n20=evaluate(model,lte,meta['num_items'],device)
    print({'HR@10':h10,'NDCG@10':n10,'HR@20':h20,'NDCG@20':n20})
if __name__=='__main__':
    main()
