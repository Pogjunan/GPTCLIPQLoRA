import os, argparse, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import GPT2Config, GPT2Mini

try:
    import sentencepiece as spm
except Exception as e:
    spm = None

class TextDataset(Dataset):
    def __init__(self, ids, ctx):
        # Create overlapping sequences of length (ctx+1) for next-token prediction
        self.ctx = ctx
        total = (len(ids) - 1) // ctx
        self.inputs = np.array(ids[: total * ctx], dtype=np.int64).reshape(total, ctx)
        self.labels = np.array(ids[1: total * ctx + 1], dtype=np.int64).reshape(total, ctx)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.labels[idx])

def load_text_and_encode(path, sp_model_path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if sp_model_path is None:
        raise ValueError("Provide --sp_model (SentencePiece model path). Train one in tokenizer/.")
    if spm is None:
        raise ImportError("sentencepiece not installed")
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    ids = sp.encode(text, out_type=int)
    return ids, sp

def save_ckpt(model, opt, step, out_dir, sp_path):
    os.makedirs(out_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "sp_model": sp_path,
    }
    torch.save(state, os.path.join(out_dir, "ckpt_last.pt"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/tiny.txt')
    ap.add_argument('--sp_model', type=str, default='tokenizer/spm.model')
    ap.add_argument('--out_dir', type=str, default='runs/gpt2mini_tiny')
    ap.add_argument('--vocab_size', type=int, default=50000)
    ap.add_argument('--n_ctx', type=int, default=512)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--n_layer', type=int, default=12)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--max_steps', type=int, default=1000)
    ap.add_argument('--log_interval', type=int, default=50)
    ap.add_argument('--bf16', action='store_true')
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ids, sp = load_text_and_encode(args.data, args.sp_model)
    cfg = GPT2Config(vocab_size=args.vocab_size, n_ctx=args.n_ctx, d_model=args.d_model,
                     n_layer=args.n_layer, n_head=args.n_head, dropout=args.dropout)
    model = GPT2Mini(cfg).to(device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=not args.bf16)
    model.train()

    ds = TextDataset(ids, ctx=args.n_ctx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # Training loop
    step, t0 = 0, time.time()
    while step < args.max_steps:
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
                logits, loss = model(x, labels=x)  # labels=x (next-token)
            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step % args.log_interval == 0:
                dt = time.time() - t0
                print(f"step {step:6d} | loss {loss.item():.4f} | {dt:.2f}s")
                t0 = time.time()
            step += 1
            if step >= args.max_steps:
                break

    save_ckpt(model, opt, step, args.out_dir, args.sp_model)
    print(f"Saved checkpoint to {args.out_dir}/ckpt_last.pt")

if __name__ == "__main__":
    main()
