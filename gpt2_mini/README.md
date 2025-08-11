# gpt2_mini
From-scratch **GPT-2 mini**: tokenizer → model → train → sample → eval.

## Day 1 (you are here)
1. **Train BPE tokenizer** on a small text (tiny Shakespeare or your custom text).
2. Implement **GPT2Block** (pre-LN, MHA, MLP, residual, absolute pos).
3. Run **sanity overfit** on a tiny dataset to validate the training loop.

## Recommended quickstart
```bash
# 1) Train tokenizer (builds tokenizer/spm.model)
python tokenizer/train_tokenizer.py --input data/tiny.txt --vocab_size 50000 --model_prefix tokenizer/spm

# 2) Train tiny model on tiny data (sanity overfit)
bash run.sh

# 3) Sample from a checkpoint
python sampler.py --ckpt runs/gpt2mini_tiny/ckpt_last.pt --prompt "To be, or not to be"
```

### Notes
- Use **AMP/BF16**, **FlashAttention** (if installed), and **activation checkpointing** to keep memory low.
- Weight tying: `lm_head.weight = token_embedding.weight`.
- GPT-2 positions: **learned absolute** embeddings for simplicity.
- Loss: next-token cross-entropy with label shift.
