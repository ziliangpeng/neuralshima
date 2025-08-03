import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger

dv = 'mps'

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, device=dv)
        self.n = nn.Linear(dim, dim, device=dv)

    def forward(self, x):
        # return self.ln(self.n(x).relu())
        return x + self.n(self.ln(x)).relu()

    def __call__(self, x):
        return self.forward(x)

class MHAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.proj = nn.Linear(dim, 3 * dim, device=dv)
        self.proj2 = nn.Linear(dim, dim, device=dv)
        self.register_buffer('mask', torch.tril(torch.ones(1024, 1024, device=dv)).unsqueeze(0).unsqueeze(0))
        # self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (self.proj(x)).split(self.dim, dim=-1) # (B, T, 3 * dim)
        q = q.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2) # B, n_heads, T, head_dim
        k = k.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2) # B, n_heads, T, head_dim
        v = v.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2) # B, n_heads, T, head_dim

        qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        qk = qk.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(qk, dim=-1) @ v # (B, n_heads, T, head_dim)
        att = att.transpose(1, 2).reshape(B, T, D)
        return self.proj2(att)

class MQAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.proj = nn.Linear(dim, dim + 2 * self.head_dim, device=dv)
        self.proj2 = nn.Linear(dim, dim, device=dv)
        self.register_buffer('mask', torch.tril(torch.ones(1024, 1024, device=dv)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.proj(x)
        q = qkv[:, :, :self.dim]
        k = qkv[:, :, self.dim:self.dim + self.head_dim].repeat(1, 1, self.n_heads)
        v = qkv[:, :, self.dim + self.head_dim:].repeat(1, 1, self.n_heads)

        q = q.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2)

        qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        qk = qk.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(qk, dim=-1) @ v # (B, n_heads, T, head_dim)
        att = att.transpose(1, 2).reshape(B, T, D)
        return self.proj2(att)

class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.proj = nn.Linear(dim, dim + 2 * self.n_kv_heads * self.head_dim, device=dv)
        self.proj2 = nn.Linear(dim, dim, device=dv)
        self.register_buffer('mask', torch.tril(torch.ones(1024, 1024, device=dv)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.proj(x)
        q = qkv[:, :, :self.dim]
        k = qkv[:, :, self.dim:self.dim + self.n_kv_heads * self.head_dim].repeat_interleave(self.n_heads // self.n_kv_heads, dim=-1)
        v = qkv[:, :, self.dim + self.n_kv_heads * self.head_dim:].repeat_interleave(self.n_heads // self.n_kv_heads, dim=-1)

        q = q.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dim // self.n_heads).transpose(1, 2)

        qk = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        qk = qk.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(qk, dim=-1) @ v # (B, n_heads, T, head_dim)
        att = att.transpose(1, 2).reshape(B, T, D)
        return self.proj2(att)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, att_type):
        super().__init__()
        assert dim % n_heads == 0
        if att_type == 'mha':
            self.attn_type = MHAttention
        elif att_type == 'mqa':
            self.attn_type = MQAttention
        elif att_type == 'gqa':
            self.attn_type = GQAttention
        else:
            raise Exception(f'Invalid attention type: {att_type}')

        self.ln1 = nn.LayerNorm(dim, device=dv)
        self.attn = self.attn_type(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim, device=dv)
        # self.mlp = nn.FeedForward(dim, 4 * dim)
        self.proj1 = nn.Linear(dim, 4 * dim, device=dv)
        self.proj2 = nn.Linear(4 * dim, dim, device=dv)

    def forward(self, x):
        h = self.ln1(x)
        h = x + self.attn(h)
        # return self.ln2(x + self.n(x))
        h = self.ln2(h)
        return x + self.proj2(self.proj1(h).relu())
        # return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, n_layers, seq_len, attn_type):
        super().__init__()
        self.vocab_size = vocab_size
        # Convert tensors to nn.Parameter so they get tracked by the optimizer
        self.pos_emb = nn.Parameter(torch.randn(seq_len, dim, device=dv))
        self.tok_emb = nn.Parameter(torch.randn(vocab_size, dim, device=dv))
        # Use ModuleList instead of regular list so parameters are registered
        # self.blocks = nn.ModuleList([Block(dim) for i in range(n_layers)])
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, attn_type) for i in range(n_layers)])
        self.ln = nn.LayerNorm(dim, device=dv)
        self.proj = nn.Linear(dim, vocab_size, device=dv)

    def cross_entropy(self, logits, label, num_classes):
        target = F.one_hot(label, num_classes).to(dv)

        # logits_softmax = F.softmax(logits, dim=-1)
        # logits_log_softmax = torch.log(logits_softmax)

        logits_log_softmax = torch.log_softmax(logits, dim=-1)
        loss = -torch.sum(logits_log_softmax * target, dim=-1)
        return loss

    def forward(self, x, target):
        # x: (b, l)
        l = x.shape[1]
        # print(self.tok_emb.device)
        # print(x.device)
        te = self.tok_emb[x]
        pe = self.pos_emb[torch.arange(0, l)].unsqueeze(0)

        emb = te + pe
        for h in self.blocks:
            emb = h(emb)

        emb = self.ln(emb)
        logits = self.proj(emb)
        loss = self.cross_entropy(logits, target, self.vocab_size)
        return logits, loss

    def __call__(self, x, target):
        return self.forward(x, target)


class Data:
    def __init__(self, digits):
        self.digits = digits
        self.char2tok = {}
        for i in range(10):
            self.char2tok[str(i)] = i
        self.char2tok.update({
            '+': 10,
            '=': 11,
            ' ': 12,
        })
        self.tok2char = {v: k for k, v in self.char2tok.items()}

    def vocab_size(self):
        return len(self.char2tok)

    def recover(self, tensors):
        # print(tensors)
        # return ''.join([self.tok2char[t] for t in tosk])
        return [''.join([self.tok2char[t.item()] for t in tsr]) for tsr in tensors]


    def next(self):
        a = random.randint(10 ** (self.digits-1), 10 ** self.digits - 1)
        b = random.randint(10 ** (self.digits-1), 10 ** self.digits - 1)
        s = f'{a} + {b} = {a + b:06d}'
        return [self.char2tok[c] for c in s]

    def next_batch(self, batch_size):
        inputs = []
        targets = []
        for i in range(batch_size):
            d = self.next()
            inputs.append(torch.tensor(d[:-1]).to(dv))
            targets.append(torch.tensor(d[1:]).to(dv))
        input = torch.stack(tuple(inputs), dim=0)
        target = torch.stack(tuple(targets), dim=0)

        return input, target


def main():
    digits = 4
    batch_size = 4

    dataloader = Data(digits)
    # vocab_size, dim, n_heads, n_layers, seq_len):
    gmh = GPT(dataloader.vocab_size(), 512, 8, 8, digits*2 + 3 + 3 + 10, 'mha')
    ggq = GPT(dataloader.vocab_size(), 512, 8, 8, digits*2 + 3 + 3 + 10, 'gqa')
    gmq = GPT(dataloader.vocab_size(), 512, 8, 8, digits*2 + 3 + 3 + 10, 'mqa')
    # g = GPT(dataloader.vocab_size(), 1024, 8, 16, 20)
    gmh = gmh.to(dv)
    ggq = ggq.to(dv)
    gmq = gmq.to(dv)

    # print(len(list(g.parameters())))
    # for p in g.parameters():
    #     print('--')
    #     print(p.shape)
    optmh = torch.optim.Adam(params=list(gmh.parameters()), lr=3e-4)
    optgq = torch.optim.Adam(params=list(ggq.parameters()), lr=3e-4)
    optmq = torch.optim.Adam(params=list(gmq.parameters()), lr=3e-4)
    for it in range(1000000000):
        input, target = dataloader.next_batch(batch_size)
        optmh.zero_grad()
        optgq.zero_grad()
        optmq.zero_grad()
        logitsmh, lossmh = gmh(input, target)
        logitsgq, lossgq = ggq(input, target)
        logitsmq, lossmq = gmq(input, target)
        if it % 100 == 0:
            # print(loss)
            with torch.no_grad():
                logger.info(f'total loss ({it}): {torch.sum(lossmh[:, digits*2+3:]):.9f}')
                logger.info(f'total loss ({it}): {torch.sum(lossgq[:, digits*2+3:]):.9f}')
                logger.info(f'total loss ({it}): {torch.sum(lossmq[:, digits*2+3:]):.9f}')
            # print(torch.argmax(logits, dim=-1))
            # logger.info(f'pred: {dataloader.recover(torch.argmax(logits, dim=-1))}')
            # logger.info(f'true: {dataloader.recover(target)}')
            # print(g.blocks[0].attn.proj.weight[0][:4])
        lossmh = lossmh[:, digits*2+3:]
        lossgq = lossgq[:, digits*2+3:]
        lossmq = lossmq[:, digits*2+3:]
        mean_lossmh = torch.mean(lossmh)
        mean_lossgq = torch.mean(lossgq)
        mean_lossmq = torch.mean(lossmq)
        mean_lossmh.backward()
        mean_lossgq.backward()
        mean_lossmq.backward()
        optmh.step()
        optgq.step()
        optmq.step()
    # logits, loss = g(input, target)
    # print(torch.argmax(logits, dim=-1))

main()
