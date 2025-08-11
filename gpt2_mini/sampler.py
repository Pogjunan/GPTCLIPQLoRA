import argparse, torch
from model import GPT2Config, GPT2Mini
import sentencepiece as spm

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64, temperature=1.0, top_k=0, top_p=0.0):
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -model.cfg.n_ctx:]
        logits, _ = model(input_cond)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--sp_model', type=str, default=None, help="defaults to value stored in checkpoint")
    ap.add_argument('--prompt', type=str, default="Hello")
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--top_p', type=float, default=0.0)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPT2Config()  # for mini we assume defaults
    model = GPT2Mini(cfg)
    model.load_state_dict(ckpt['model'])
    model.cuda()

    sp_path = args.sp_model or ckpt.get('sp_model', None)
    sp = spm.SentencePieceProcessor(model_file=sp_path)

    ids = sp.encode(args.prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long)

    out = generate(model, input_ids, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    text = sp.decode(out[0].tolist())
    print(text)

if __name__ == "__main__":
    main()
