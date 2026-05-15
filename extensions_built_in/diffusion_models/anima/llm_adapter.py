import torch
from torch import nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def rotate_half(x):
    left, right = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].to(x.device)
        freqs = (inv_freq.float() @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        query_shape = (*x.shape[:-1], self.n_heads, self.head_dim)
        key_shape = (*context.shape[:-1], self.n_heads, self.head_dim)
        query = self.q_norm(self.q_proj(x).view(query_shape)).transpose(1, 2)
        key = self.k_norm(self.k_proj(context).view(key_shape)).transpose(1, 2)
        value = self.v_proj(context).view(key_shape).transpose(1, 2)
        if position_embeddings is not None:
            query = apply_rotary_pos_emb(query, *position_embeddings)
            key = apply_rotary_pos_emb(key, *position_embeddings_context)
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(*x.shape[:-1], -1).contiguous()
        return self.o_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.norm_self_attn = nn.RMSNorm(model_dim, eps=1e-6)
            self.self_attn = Attention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = nn.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = Attention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = nn.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None, position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            x = x + self.self_attn(
                self.norm_self_attn(x),
                mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
        x = x + self.cross_attn(
            self.norm_cross_attn(x),
            mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        return x + self.mlp(self.norm_mlp(x))


class AnimaLLMAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        source_dim: int = 1024,
        target_dim: int = 1024,
        model_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        vocab_size: int = 32128,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, target_dim)
        self.in_proj = nn.Linear(target_dim, model_dim) if model_dim != target_dim else nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [TransformerBlock(source_dim, model_dim, num_heads, mlp_ratio, use_self_attn) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = nn.RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        if target_attention_mask is not None and target_attention_mask.ndim == 2:
            target_attention_mask = target_attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)
        if source_attention_mask is not None and source_attention_mask.ndim == 2:
            source_attention_mask = source_attention_mask.to(torch.bool).unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        context_position_ids = torch.arange(source_hidden_states.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        context_position_embeddings = self.rotary_emb(x, context_position_ids)
        for block in self.blocks:
            x = block(
                x,
                source_hidden_states,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=context_position_embeddings,
            )
        return self.norm(self.out_proj(x))
