import torch
import inspect
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()

        self.n_head = n_head
        self.n_embed = n_embed

        assert self.n_embed % self.n_head == 0, "Embedding dimension must be divisible by number of heads"

        self.c_attention = nn.Linear(self.n_embed, 3 * self.n_embed) # for all 3 (key, query, value)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attention(x)
        q, k, v = qkv.split(self.n_embed, dim = 2) # 3 -> (B, T, n_embed)

        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Reshape back to (B, T, n_embed)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.n_embed = n_embed

        self.input_hidden = nn.Linear(self.n_embed, 4 * self.n_embed)
        self.gelu = nn.GELU()
        self.hidden_output = nn.Linear(4 * self.n_embed, self.n_embed)
        self.hidden_output.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.input_hidden(x)
        x = self.gelu(x)
        x = self.hidden_output(x)
        return x


class Block(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()

        self.n_embed = hyperparameters.n_embed
        self.n_head = hyperparameters.n_head

        head_size = self.n_embed // self.n_head
        self.ln1 = nn.LayerNorm(self.n_embed)
        self.attention = CasualSelfAttention(self.n_head, self.n_embed)
        self.ln2 = nn.LayerNorm(self.n_embed)
        self.mlp = MLP(self.n_embed)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, hyperparameters, master_process = False):
        super().__init__()
        
        self.master_process = master_process

        self.vocab_size = hyperparameters.vocab_size
        self.n_embed = hyperparameters.n_embed
        self.n_layer = hyperparameters.n_layer
        self.block_size = hyperparameters.block_size

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embed)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embed)
        self.transformer_blocks = nn.Sequential(*[Block(hyperparameters) for _ in range(self.n_layer)])
        self.final_layer_norm = nn.LayerNorm(self.n_embed)
        self.language_model_head = nn.Linear(self.n_embed, self.vocab_size, bias = False)
        self.language_model_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layer) ** -0.5 # 2 -> attention and mlp blocks

            torch.nn.init.normal_(module.weight, mean=0.0, std = std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)


    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        token_embeddings = self.token_embedding_table(idx) #(Batch, Time, Channel)
        position_embeddings = self.position_embedding_table(torch.arange(0, T, dtype = torch.long, device = idx.device)) #(Time, Channel)
        x = token_embeddings + position_embeddings

        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)

        logits = self.language_model_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    # from github ->
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Get all trainable parameters (that require gradients)
        trainable_params = {name: param for name, param in self.named_parameters() if param.requires_grad}

        # Separate parameters based on weight decay application
        weight_decay_params = [param for name, param in trainable_params.items() if param.dim() >= 2]
        no_weight_decay_params = [param for name, param in trainable_params.items() if param.dim() < 2]

        # Define parameter groups for optimizer
        param_groups = [
            {'params': weight_decay_params, 'weight_decay': weight_decay},
            {'params': no_weight_decay_params, 'weight_decay': 0.0}  # No weight decay for biases and LayerNorm
        ]

        # Count the number of parameters in each group (for debugging)
        num_weight_decay_params = sum(param.numel() for param in weight_decay_params)
        num_no_weight_decay_params = sum(param.numel() for param in no_weight_decay_params)

        if self.master_process:
            print(f"Weight-decayed parameter tensors: {len(weight_decay_params)}, total parameters: {num_weight_decay_params:,}")
            print(f"Non-weight-decayed parameter tensors: {len(no_weight_decay_params)}, total parameters: {num_no_weight_decay_params:,}")

        # Check if fused AdamW is available for GPU acceleration
        fused_adamw_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused_adamw = fused_adamw_available and device_type == "cuda"

        if self.master_process:
            print(f"Using fused AdamW: {use_fused_adamw}")

        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            lr = learning_rate,
            betas = (0.9, 0.95),
            eps = 1e-8,
            fused = use_fused_adamw
        )

        return optimizer


    # from github ->
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, do_sample = False, top_k = None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
