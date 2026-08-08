import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SequenceAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for sequence history compression.
    This is designed to be integrated into the policy network for learning.
    """
    def __init__(self, hidden_dim=1024, num_heads=4, dropout=0.1, weight_mode='learned'):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.weight_mode = weight_mode
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, current_embed, history_embeds):
        """
        Args:
            current_embed: [batch_size, hidden_dim] - current step embedding
            history_embeds: [batch_size, seq_len, hidden_dim] - history sequence
            
        Returns:
            attended_output: [batch_size, hidden_dim] - weighted sum of history
            attention_weights: [batch_size, num_heads, 1, seq_len] - attention weights
        """
        batch_size, seq_len = history_embeds.shape[:2]
        
        # Project to query, key, value
        q = self.q_proj(current_embed).unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(history_embeds)  # [B, seq_len, D]
        v = self.v_proj(history_embeds)  # [B, seq_len, D]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, seq_len, d]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, seq_len, d]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, 1, seq_len]
        attn_weights = self._make_attention_weights(attn_scores)
        self.last_attention_weights_raw = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, 1, d]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)  # [B, 1, D]
        attended_output = self.out_proj(attn_output.squeeze(1))  # [B, D]
        
        return attended_output, attn_weights

    def _make_attention_weights(self, attn_scores):
        mode = getattr(self, 'weight_mode', 'learned')
        if mode in ('learned', 'normal'):
            return F.softmax(attn_scores, dim=-1)
        if mode == 'uniform':
            return torch.full_like(attn_scores, 1.0 / attn_scores.size(-1))
        if mode == 'random':
            return F.softmax(torch.randn_like(attn_scores), dim=-1)
        raise ValueError(f"Unknown attention weight mode: {mode}")


class SimpleSequenceAttention(nn.Module):
    """
    Single-head attention pooling for sequence history compression.
    Simpler alternative to multi-head attention.
    """
    def __init__(self, hidden_dim=1024, dropout=0.1, weight_mode='learned'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.weight_mode = weight_mode
        
        # Scale factor
        self.scale = 1.0 / math.sqrt(hidden_dim)
        
    def forward(self, current_embed, history_embeds):
        """
        Args:
            current_embed: [batch_size, hidden_dim] - current step embedding
            history_embeds: [batch_size, seq_len, hidden_dim] - history sequence
            
        Returns:
            attended_output: [batch_size, hidden_dim] - weighted sum of history
            attention_weights: [batch_size, 1, seq_len] - attention weights
        """
        batch_size = current_embed.shape[0]
        
        # Project to query, key, value
        q = self.q_proj(current_embed).unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(history_embeds)  # [B, seq_len, D]
        v = self.v_proj(history_embeds)  # [B, seq_len, D]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, 1, seq_len]
        attn_weights = self._make_attention_weights(attn_scores)
        self.last_attention_weights_raw = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, 1, D]
        attended_output = self.out_proj(attn_output.squeeze(1))  # [B, D]
        
        return attended_output, attn_weights

    def _make_attention_weights(self, attn_scores):
        mode = getattr(self, 'weight_mode', 'learned')
        if mode in ('learned', 'normal'):
            return F.softmax(attn_scores, dim=-1)
        if mode == 'uniform':
            return torch.full_like(attn_scores, 1.0 / attn_scores.size(-1))
        if mode == 'random':
            return F.softmax(torch.randn_like(attn_scores), dim=-1)
        raise ValueError(f"Unknown attention weight mode: {mode}")


def create_sequence_attention(attention_type='none', hidden_dim=1024, num_heads=4, dropout=0.1, weight_mode='learned'):
    """
    Factory function to create sequence attention module.
    
    Args:
        attention_type: 'multi_head'/'multihead', 'simple', or 'none'
        hidden_dim: embedding dimension
        num_heads: number of attention heads (for multihead)
        dropout: dropout rate
        
    Returns:
        Attention module or None
    """
    if attention_type in ('multi_head', 'multihead'):
        return SequenceAttentionPooling(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, weight_mode=weight_mode)
    elif attention_type == 'simple':
        return SimpleSequenceAttention(hidden_dim=hidden_dim, dropout=dropout, weight_mode=weight_mode)
    elif attention_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
