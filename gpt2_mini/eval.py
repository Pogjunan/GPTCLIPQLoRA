import argparse, math, torch, sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from model import GPT2Config, GPT2Mini

class EvalDataset(Dataset):
    def __init__(self, ids, ctx):
        self.ctx = ctx
        total = (len(ids) - 1) // ctx
        self.inputs = torch.tensor(ids[: total * ctx], dtype=torch.long).view(total, ctx)
    def __len__(self):
        return self.inputs.size(0)
    def __getitem__(self, idx):
        x = self.inputs[idx]
        return x, x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--sp_model', type=str, required=True)
    ap.add_argument('--n_ctx', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    with open(args.data, 'r', encoding='utf-8') as f:
        ids = sp.encode(f.read(), out_type=int)

    ds = EvalDataset(ids, ctx=args.n_ctx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    cfg = GPT2Config(n_ctx=args.n_ctx)
    model = GPT2Mini(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
    model.cuda().eval()

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.cuda()
            logits, loss = model(x, labels=x)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    ppl = math.exp(total_loss / total_tokens)
    print(f"Perplexity: {ppl:.3f}")

if __name__ == '__main__':
    main()
