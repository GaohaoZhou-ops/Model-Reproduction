# 1. 导入必要的包
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

#-----------------------------------------------------------------------#
# 2. 定义归一化层类
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
#-----------------------------------------------------------------------#
# 3. 定义 CausalSelfAttention 类
class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0   # 确保嵌入维度（n_embd）能被注意力头数（n_head）整除，每个头将分配到相同大小的子空间
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_head, bias=config.bias)   # 线性变换层，用于生成查询（Q）、键（K）和值（V），将嵌入维度扩展为三倍，以便后续拆分为 Q、K、V。
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)     # 将注意力输出重新投影回原始维度
        self.attn_dropout = nn.Dropout(config.dropout)      # 用于注意力矩阵（防止过拟合）
        self.resid_dropout = nn.Dropout(config.dropout)     # 用于残差连接的输出
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropput
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:  # 判断是否支持GPU 加速的Flash Attention
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 使用下三角矩阵构建因果掩码，确保只关注左侧信息（即历史信息）
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))
            
    def forward(self, x):
        B,T,C = x.size()    # batch_size, token_length, embedding_dim
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)    # 计算 Q、K、V
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # shape=(B,nh,T,hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # shape=(B,nh,T,hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)  # shape=(B,nh,T,hs)
        
        if self.flash:      # 如果支持GPU加速，则使用flash attention
            y = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:               # 否则手动计算
            att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))    
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v     # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)       # 还原为原始形状，便于后续处理
        
        y = self.resid_dropout(self.c_proj(y))
        return y
    
#-----------------------------------------------------------------------#
# 4. 定义 MLP 类
class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    
#-----------------------------------------------------------------------#
# 5. 定义单个计算单元
class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
#-----------------------------------------------------------------------#
# 6. 定义GPT类需要的配置对象
@dataclass
class GPTConfig:
    block_size: int=1024
    vocab_size: int=50304
    n_layer: int=12
    n_head: int=12
    n_embd: int=768
    dropout: float=0.0
    bias: bool=True
    
#-----------------------------------------------------------------------#
# 7. 定义完整GPT模型
class GPT(nn.Module):
    def __init__(self, config:GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None    # 检查配置中 词汇表大小（vocab_size）和块大小（block_size）是否有效。
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(                      # Transformer结构
            wte=nn.Embedding(config.vocab_size, config.n_embd),     # 词嵌入（wte）：将词索引映射为向量表示，shape=(vocab_size, n_embd)
            wpe=nn.Embedding(config.block_size, config.n_embd),     # 位置嵌入（wpe）：将位置索引映射为向量，shape=(block_size, n_embd)
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),    # 多个并列的注意力块，每个块由自定义 Block(config) 生成
            ln_f = LayerNorm(config.n_embd, bias=config.bias),      # 最终输出的层归一化
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言模型头：将最后的特征映射回词汇表空间，用于词预测
        self.transformer.wte.weight = self.lm_head.weight   # 权重绑定：词嵌入层和输出投影层共享权重，减少参数量，提高性能
        
        self.apply(self._init_weights)      # 权重初始化
        for pn, p in self.named_parameters():   # 对残差连接权重（c_proj）进行特殊初始化，符合 GPT-2 论文规范
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    # 获得参数总量
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    # 初始化整体权重
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        device = idx.device
        b,t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"   # 确保输入序列长度不超过配置的块大小
        pos = torch.arange(0, t, dtype=torch.long, device=device)   # 位置编码，shape=(t)
        
        # 嵌入层前向计算：词嵌入 & 位置嵌入
        tok_emb = self.transformer.wte(idx) # shape=(b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # shape=(t, n_embd)
        x = self.transformer.drop(tok_emb+pos_emb)
        
        # 逐层通过多层注意力块（Block），最后进行层归一化
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:         # 训练阶段
            logits = self.lm_head(x)        # 计算预测分布（logits）：使用语言模型头得到词概率分布
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # 交叉熵损失：比较预测值和目标值，忽略填充索引 -1
        else:                           # 推理阶段
            logits = self.lm_head(x[:, [-1], :])    # 仅计算最后一个时间步的预测，减少推理时的计算量
            loss = None
        return logits, loss
    
    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
        
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k=='dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # 模型配置参数
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        
        # 如果 dropout 在参数列表中则对齐进行覆写
        if 'dropout' in override_args:            
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        # 加载预训练权重
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # 因为这里是nanoGPT，所以只需要从完整GPT模型中拷贝一部分权重即可
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
       
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model            
        
    # 配置 AdamW 优化器
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
    
    # 估算模型的GPU利用率
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L,H,Q,T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.1, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx [:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx