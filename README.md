# GPTCLIPQLoRA (Project Scaffold)
This repository hosts a 6-week learning-by-building plan:
- `gpt2_mini/`: **From-scratch GPT-2 mini** (tokenizer → model → trainer → sampler → eval).
- `mini_clip/`: Minimal CLIP (contrastive I↔T) with ablations.
- `mm_instruct/`: BLIP-2/LLaVA-lite style vision–language instruction pipeline.
- `rag_mvp/`: RAG pipeline (embedding → HNSW → rerank → LLM) with citations.
- `moe_playground/`: Tiny MoE insertion + routing experiments.
- `train_scaling/`: FSDP/ZeRO, FlashAttention, checkpointing experiments.
- `demo/`: End-to-end demo (Week 6).
- `common/`: Utilities shared across modules.
- `reports/`: Figures and tables.

> Day 1 focus: `gpt2_mini/` tokenizer + model skeleton + sanity-overfit on tiny data.
