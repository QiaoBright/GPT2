from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class GPTConfig:
    block_size: int=1024
    vocab_size:int=50257
    n_layer:int=12
    n_head:int=12
    n_embd:int=768

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    
    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        # att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.glue = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.glue(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x+ self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self,module):
        
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std = (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        B,T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))

        return logits,loss

    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt2 {model_type}")

        config_args = {
            'gpt2':{'n_layer':12,'n_head':12,'n_embd':768},
            'gpt2-medium':{'n_layer':24,'n_head':16,'n_embd':1024},
            'gpt2-large':{'n_layer':36,'n_head':20,'n_embd':1280},
            'gpt2-xl':{'n_layer':48,'n_head':25,'n_embd':1600},
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() 
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with  torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
from torch.distributed import init_process_group
import os
ddp = int(os.environ.get('RANK',-1)) != -1
if ddp:
    assert torch.cuda.is_available(),"for now i think we need cuda for ddp"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device {device}")
# device = "cpu"


import tiktoken
# model = GPT.from_pretrained('gpt2')

class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('/usr/3Tusr/qiaojiaqi/llm/GPT2/input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        self.current_postion = self.B * self.T * self.process_rank

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_postion:self.current_postion+B*T+1]
        x =  (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_postion += B * T * self.num_processes
        if self.current_postion + (B*T*self.num_processes+1) >= len(self.tokens):
            self.current_postion = 0
        return x,y


torch.set_float32_matmul_precision('high')
# 很多工作负载是受限于内存的，虽然TF32原则上提供了更快的吞吐量，但这些数字仍然是float32,这是float32的数字在内存系统中不断地传输。
#尽管我们使乘法本身变得更快，我们还是受限于内存，并没有看到完全的收益
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B=16
T=1024
train_loader = DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size)
assert total_batch_size % (B*T*ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size:{total_batch_size}")
    print(f"=> calculated gradient accmulation steps:{grad_accum_steps}")


model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# 加速主要来自减少Python开销和GPU读写
model = torch.compile(model)
if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > warmup_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = torch.optim .AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)
# optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device),y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss = model(x,y)
            # import code;code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1-t0)
    if master_process:
        print(f"step{i} | loss:{loss_accum.item():.6f} | lr:{lr:.4f} | dt:{dt:.2f}ms | norm:{norm:. 4f} | tok/sec:{tokens_per_sec:.2f}")

if ddp:
    dict.destory_process_group()
# logits,loss = model(x,y)
# print(loss)
# import sys
# sys.exit()

# model = GPT(GPTConfig())
# num_return_sequences = 5
# max_length = 30
# model.eval()
# model.to('cuda')



# tokens = enc.encode("Hello, my dog is cute")
# tokens = torch.tensor(tokens,dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
# x = tokens.to('cuda')

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     logits = model(x)
#     logits = logits[:,-1,:]
#     probs = F.softmax(logits,dim=-1)
#     topk_probs, topk_indices = torch.topk(probs,50,dim=-1)
#     ix = torch.multinomial(topk_probs,num_samples=1)
#     xcol = torch.gather(topk_indices,-1,ix)
#     x = torch.cat((x,xcol),dim=-1)

# for i in range(num_return_sequences):
#     tokens = x[i,:max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">. ",decoded)

