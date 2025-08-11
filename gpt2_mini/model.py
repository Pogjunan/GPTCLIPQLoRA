import math, torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Config:
    def __init__(self, vocab_size=50000, n_ctx=512, d_model=256, n_layer=12, n_head=8, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.view(B, T, 3, self.n_head, self.head_dim).permute(2,0,3,1,4)
        # q,k,v: (B, n_head, T, head_dim)
        # Use PyTorch 2.0 scaled_dot_product_attention for speed; causal via is_causal=True
        # attn_mask is not required if we set is_causal=True.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.out(y)
        return self.resid_drop(y)

class GPT2Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT2Mini(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.n_ctx, cfg.d_model)  # learned absolute positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPT2Block(cfg.d_model, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        assert T <= self.cfg.n_ctx, "Sequence length exceeds model context window."
        pos = torch.arange(0, T, device=input_ids.device, dtype=torch.long).unsqueeze(0)  # (1, T)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                                   labels[:, 1:].contiguous().view(-1))
        return logits, loss
