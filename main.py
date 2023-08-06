import torch.jit as jit
from typing import Any, Callable
import numpy as np
import torch.nn.functional as fn
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List
from dataclasses import dataclass, asdict
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence
import csv
import datetime
import os
from lion_pytorch import Lion
import random
import pandas as pd
import regex
from tqdm import tqdm
import transformers
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
max_seq_len = 256


class Projection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        order: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 2
    ):
        super().__init__()
        hidden_size = (order + 1) * embed_dim
        self.linear = nn.Linear(embed_dim, hidden_size)
        self.short_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_size
        )
        self.hidden_size = hidden_size

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        args
        - `x`: input tensor with shape (batches, length, embed dim)
        """
        # B: batch size, L: seq len, E: embed dim, N: order of hyena
        L = x.shape[1]
        x = (
            self.linear(x)  # (B, L, E) -> (B, L, (N+1)*E)
            .transpose(1, 2)  # (B, L, (N+1)*E) -> (B, (N+1)*E, L)
        )
        x = self.short_conv(x)[..., :L]  # (B, (N+1)*E, L) -> (B, (N+1)*E, L)
        # (B, (N+1)*E, L) -> [(B, E, L)] * (N+1)
        return x.chunk(self.hidden_size, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: int):
        assert embed_dim % 2 == 1, "`embed_dim` must be odd"
        super().__init__()
        # L: seq len, Ep: pos embed dim, K: (Et-1)//2
        t = torch.linspace(0, 1, steps=max_seq_len).unsqueeze(-1)  # -> (L, 1)
        t_pos = torch.arange(
            0, max_seq_len, dtype=torch.float).unsqueeze(-1)  # -> (L, 1)
        K = (embed_dim - 1) // 2
        k = torch.linspace(0, K - 1, steps=K).unsqueeze(0)  # -> (1, K)
        z = torch.exp(1j * 2 * np.pi * k * t_pos / max_seq_len)  # -> (L, K)
        self.t = nn.Parameter(t.view(1, 1, max_seq_len),
                              requires_grad=False)  # -> (1, 1, L)
        self.z = nn.Parameter(
            torch.cat([t, z.real, z.imag], dim=-1),  # -> (L, Ep)
        )

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.t[..., :seq_len], self.z[:seq_len, :]


class Sin(nn.Module):
    def __init__(self, embed_dim: int, freq: float = 8.0, learn: bool = True):
        super().__init__()
        self.freq = nn.Parameter(
            freq * torch.ones(1, embed_dim), requires_grad=learn)

    def forward(self, x: Tensor) -> Tensor:
        # L: seq len, E: embed dim
        return torch.sin(self.freq * x)  # -> (L, E)


class ExponentialDecayWindow(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        fast_decay_t: float = 0.3,
        slow_decay_t: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0
    ):
        super().__init__()
        max_decay = np.log(target) / fast_decay_t
        min_decay = np.log(target) / slow_decay_t
        self.alphas = nn.Parameter(
            torch.linspace(min_decay, max_decay,
                           steps=embed_dim).view(1, embed_dim, 1)
        )
        self.shift = shift

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # L: seq len, E: embed dim, N: order of hyena
        L = x.shape[-1]
        decay = torch.exp(self.alphas * t)[..., :L]  # -> (1, E, L)
        x *= (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
        self,
        pos_embed_dim: int,
        max_seq_len: int,
        seq_embed_dim: int,
        order: int = 2,
        fnn_depth: int = 4,
        fnn_hidden_size: int = 64,
        freq: float = 10.0,
        learn: bool = True,
        fast_decay_t: float = 0.3,
        slow_decay_t: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0
    ):
        assert fnn_depth > 2, "`fnn_depth` must be greater than 2"
        super().__init__()
        self.pos = PositionalEncoding(pos_embed_dim, max_seq_len)

        self.fnn = nn.Sequential(
            nn.Linear(pos_embed_dim, fnn_hidden_size),
            Sin(fnn_hidden_size, freq, learn)
        )
        for _ in range(fnn_depth - 2):
            self.fnn.append(nn.Linear(fnn_hidden_size, fnn_hidden_size))
            self.fnn.append(Sin(fnn_hidden_size, freq, learn))
        self.fnn.append(nn.Linear(fnn_hidden_size,
                        order * seq_embed_dim, bias=False))

        self.embed_dim = seq_embed_dim
        self.order = order
        self.window = ExponentialDecayWindow(
            seq_embed_dim,
            fast_decay_t=fast_decay_t,
            slow_decay_t=slow_decay_t,
            target=target,
            shift=shift
        )

    def forward(self, seq_len: int) -> list[Tensor]:
        # L: seq len, Ep: pos embed dim, N: order of hyena, E: seq embed dim
        t, z = self.pos(seq_len)  # -> (1, 1, L), (L, Ep)
        h = (
            self.fnn(z)  # (L, Ep) -> (L, N*E)
            .transpose(0, 1)  # (L, N*E) -> (N*E, L)
            # (N*E, L) -> (N, E, L)
            .reshape(self.order, self.embed_dim, seq_len)
        )
        h = self.window(h, t)  # (N, E, L) -> (N, E, L)
        return h.chunk(self.order, dim=0)  # (N, E, L) -> [(1, E, L)] * N


class HyenaBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        max_seq_len: int,
        order: int,
        pos_dim: int = 65,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 2,
        fnn_depth: int = 4,
        fnn_hidden_size: int = 64,
        freq: float = 8.0,
        learn_filter: bool = True,
        fast_decay_t: float = 0.3,
        slow_decay_t: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0,
        activation: str = "identity"
    ):
        super().__init__()
        self.proj = Projection(
            embed_dim,
            order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.hyena_filter = HyenaFilter(
            pos_dim,
            max_seq_len,
            seq_embed_dim=embed_dim,
            order=order,
            fnn_depth=fnn_depth,
            fnn_hidden_size=fnn_hidden_size,
            freq=freq,
            learn=learn_filter,
            fast_decay_t=fast_decay_t,
            slow_decay_t=slow_decay_t,
            target=target,
            shift=shift
        )
        self.bias = nn.Parameter(torch.randn(order, 1, embed_dim, 1))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

        act: nn.Module
        match name := activation.lower():
            case "identity": act = nn.Identity()
            case "relu": act = nn.ReLU()
            case "leaky-relu": act = nn.LeakyReLU()
            case "gelu": act = nn.GELU()
            case "silu": act = nn.SiLU()
            case "tanh": act = nn.Tanh()
            case _: raise NotImplementedError(f"activation `{name}` is invalid")
        self.act = act

    @staticmethod
    def fftconv(x: Tensor, h: Tensor, d: Tensor) -> Tensor:
        # B: batch size, L: seq len, E: embed dim
        L = x.shape[-1]
        # (1, E, L) -> (1, E, 2*L)
        h_fft = torch.fft.rfft(h.float(), n=2*L, norm="forward")
        # (B, E, L) -> (B, E, 2*L)
        x_fft = torch.fft.rfft(x.float(), n=2*L)
        y = torch.fft.irfft(x_fft * h_fft, n=2*L,
                            norm="forward")[..., :L].bfloat16()  # -> (B, E, L)
        y += x * d
        return y.to(dtype=x.dtype)

    def forward(self, u: Tensor) -> Tensor:
        # B: batch size, L: seq len, E: embed dim, N: order of hyena
        L = u.shape[1]
        x = self.norm1(u)  # (B, L, E) -> (B, L, E)
        x = self.proj(x)  # (B, L, E) -> [(B, E, L)] * (N+1)
        h = self.hyena_filter(L)  # -> [(1, E, L)] * N
        v = x[-1]  # -> (B, E, L)
        for x_i, h_i, d_i in zip(x[:-1], h, self.bias):
            v = x_i * self.fftconv(v, h_i, d_i)
        y = u + v.transpose(1, 2)  # -> (B, L, E)
        out = self.norm2(y)  # (B, L, E) -> (B, L, E)
        out = self.fc(out)  # (B, L, E) -> (B, L, E)
        out = self.act(out)  # (B, L, E) -> (B, L, E)
        out += y
        return out


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


@dataclass(frozen=False)
class HyenaConfig:
    embed_dim: int = 512
    max_seq_len: int = 256
    order: int = 2
    pos_dim: int = 65
    kernel_size: int = 3
    stride: int = 1
    padding: int = 2
    fnn_depth: int = 4
    fnn_hidden_size: int = 64
    freq: float = 8.0
    learn_filter: bool = True
    fast_decay_t: float = 0.3
    slow_decay_t: float = 1.5
    target: float = 1e-2
    shift: float = 0.0
    activation: str = "identity"


class my_Hyena_head(nn.Module):
    def __init__(self, num_head, hyena_config: HyenaConfig):
        super().__init__()
        self.num_head = num_head
        self.heads = nn.ModuleList([HyenaBlock(**asdict(hyena_config))
                                    for i in range(num_head)])

    def forward(self, x: Tensor):
        a, b, c = x.size()
        # x = x.view(a*self.num_head, b, c//self.num_head)
        x = x.chunk(self.num_head, -1)

        li = []
        for i in range(self.num_head):
            li.append(self.heads[i](x[i]))
        x = torch.cat(li, dim=-1)

        return x


class HyenaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        depth: int,
        hyena_config: HyenaConfig,
        p_dropout: float = 0.0,
        pe_type: str = "absolute",
        pad_id: int | None = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        embed_dim = hyena_config.embed_dim
        max_seq_len = hyena_config.max_seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(p_dropout)

        pos_embed: Tensor
        pe_requires_grad = False
        match name := pe_type.lower():
            case "fixed":
                pos_embed = torch.zeros(1, max_seq_len, embed_dim)
                omega = 1. / \
                    (10000 ** (torch.arange(0, embed_dim, 2, dtype=torch.float) / embed_dim))
                pos = torch.arange(
                    0, max_seq_len, dtype=torch.float).unsqueeze(-1)
                theta = omega * pos
                pos_embed[..., 0::2] = torch.sin(theta)
                pos_embed[..., 1::2] = torch.cos(theta)
            case "absolute":
                pos_embed = torch.randn(1, max_seq_len, embed_dim)
                pe_requires_grad = True
            case "nope":
                pos_embed = torch.zeros(1, max_seq_len, embed_dim)
            case _: raise NotImplementedError(f"positional encoding `{name}` is invalid")
        self.pos_embed = nn.Parameter(
            pos_embed, requires_grad=pe_requires_grad)
        self.num_head = 8
        hyena_config.embed_dim //= self.num_head

        self.layers = nn.ModuleList(
            [my_Hyena_head(self.num_head, hyena_config) for _ in range(depth)]
        )

        hyena_config.embed_dim *= self.num_head

        self.norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)
        self.d = depth

    def forward(self, x: Tensor) -> Tensor:
        # B: batch size, L: seq_len, E: embed dim, V: vocab size
        x = self.embed(x)  # (B, L) -> (B, L, E)
        x += self.pos_embed[:, :x.shape[1], :]
        x = self.dropout(x)  # (B, L, E) -> (B, L, E)
        for i in range(self.d):
            x = checkpoint(self.layers[i], x)  # (B, L, E) -> (B, L, E)
        x = self.norm(x)  # (B, L, E) -> (B, L, E)
        x = self.out(x)  # (B, L, E) -> (B, L, V)
        return x


class Generator:
    def __init__(
        self,
        model: HyenaLM,
        encoder: Callable[[str], list[int]],
        decoder: Callable[[list[int]], str],
        max_seq_len: int,
        bos: int,
        eos: int,
        device: Any = torch.device("cpu")
    ):
        self.model = model.to(device=device)
        self.encoder = encoder
        self.decoder = decoder
        self.max_seq_len = max_seq_len
        self.bos = bos
        self.eos = eos
        self.device = device

    def generate(
        self,
        prompt: str,
        output_len: int,
        k: int = 10,
        temperature: float = 1.0,
        num_repeat: int = 2
    ) -> str:
        # L: seq len, V: vocab size, K: k
        self.model.eval()
        tokens = self.encoder(prompt)
        tokens.insert(0, self.bos)
        while len(tokens) < output_len + 1:
            x = torch.unsqueeze(
                torch.tensor(tokens[-self.max_seq_len:],
                             dtype=torch.long, device=self.device), 0
            )  # -> (1, L)
            logits = self.model(x)[0, -1, :]  # -> (V)

            values, indices = torch.topk(logits, k)  # -> (K) for each
            probas = torch.full_like(logits, float("-inf"))  # -> (V)
            probas.scatter_(0, indices, values)
            probas = fn.softmax(probas / temperature, dim=-1)  # (V) -> (V)

            next_token = torch.multinomial(probas, 1).item()
            # Check for repetition and if it's too much, redraw the token.
            count = 0
            for i in range(1, num_repeat + 1):
                if len(tokens) - i < 0 or tokens[-i] != next_token:
                    break
                count += 1
            while count >= num_repeat:
                next_token = torch.multinomial(probas, 1).item()
                count = 0
                for i in range(1, num_repeat + 1):
                    if len(tokens) - i < 0 or tokens[-i] != next_token:
                        break
                    count += 1

            tokens.append(next_token)
            print(self.decoder(next_token), end="")
        output = self.decoder(tokens)
        return output


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_):
        self.data = data
        self.tokenizer = tokenizer
        self.max_ = max_
        self.cash = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            self.tokenizer.sep_token_id = 5
            d = self.data[idx]

            input = self.tokenizer.encode_plus(d, truncation=True,
                                               max_length=self.max_, return_tensors="pt")  # Do not pad here

            input_ids = input["input_ids"]
            input_ids = (input_ids[0][:input_ids.size(1)-1]).unsqueeze(0)
            return {"input_ids": input_ids[0]}
            answer = f"{out}"

            input = self.tokenizer.encode_plus(question, truncation=True,
                                               max_length=self.max_, padding="max_length", return_tensors="pt")  # Do not pad here
            target_ids = self.tokenizer.encode_plus(answer, truncation=True,
                                                    max_length=self.max_, padding="max_length", return_tensors="pt")["input_ids"]  # Do not pad here

            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]

            return {"input_ids": input_ids[0], "attention_mask": attention_mask[0], "labels": target_ids[0], "text": question}
        except Exception as e:
            print(e)
            return {"input_ids": torch.zeros(self.max_length).int(), "attention_mask": torch.zeros(self.max_length).int(), "labels": torch.zeros(self.max_length).int()}


tokenizer = transformers.AutoTokenizer.from_pretrained(
    'rinna/japanese-gpt-1b', use_fast=False)
# データを整形


def split_text(text, length):
    words = text.split(' ')
    chunks = []
    chunk = []
    chunk_len = 0
    for word in words:
        if chunk_len + len(word) + 1 > length:  # The "+1" accounts for the space
            chunks.append(' '.join(chunk))
            chunk = [word]
            chunk_len = len(word)
        else:
            chunk.append(word)
            chunk_len += len(word) + 1
    chunks.append(' '.join(chunk))
    return chunks


tqdm.pandas()

embed_dim = 2048
vocab_size = tokenizer.vocab_size
depth = 24
config = HyenaConfig(
    embed_dim=embed_dim,
    max_seq_len=max_seq_len,
    activation="gelu"
)
model = HyenaLM(vocab_size, depth, hyena_config=config).cuda().bfloat16()


df_ = pd.read_parquet('text.parquet')


df_ = df_[~df_['text'].str.contains(
    '#質問|#vrchat|zabuu|#shindanmaker|#DLsite', case=False, regex=True) & ~df_['text'].str.startswith('RT ')]

# 質問箱 #VRChat
# df2 = [s for s in a if (len(s)>64 and len(s) <= 128) ]
# df = sorted(df, key=len)
df_ = list(df_["text"])
df_ = sorted(df_, key=len)
df_ = [s for s in df_ if (len(s) > 8)]
# dataset = QADataset(df, tokenizer,  max_=max_seq_len)
dataset_ = QADataset(df_, tokenizer,  max_=34)


def collate_fn(batch):

    input_ids = [item['input_ids'][:-1] for item in batch]
    input_ids2 = [item['input_ids'][1:] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=4)
    input_ids2 = pad_sequence(input_ids2, batch_first=True, padding_value=4)

    return {
        'input_ids': input_ids,
        'input_ids2': input_ids2,
        #    'attention_mask': torch.stack(padded_attention_mask).long(),
        #   'labels': torch.stack(padded_labels).long(),
    }


model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


loss_fn = nn.CrossEntropyLoss(ignore_index=4, label_smoothing=0.1)
dataloader_ = DataLoader(
    dataset_, batch_size=16*12,    collate_fn=collate_fn, shuffle=True)

# dataloader = DataLoader(
#   dataset, batch_size=40*3,    collate_fn=collate_fn,shuffle=True)
# dataloader2 = DataLoader(
#   dataset2, batch_size=16*4,    collate_fn=collate_fn)
epochs = 1


def encoder_fn(text):
    return tokenizer.encode(text, add_special_tokens=False)


def decoder_fn(tokens):
    return tokenizer.decode(tokens)


generator = Generator(
    model=model,
    encoder=encoder_fn,
    decoder=decoder_fn,
    max_seq_len=max_seq_len,
    bos=tokenizer.bos_token_id,
    eos=tokenizer.eos_token_id,
    device=torch.device("cuda") if torch.cuda.is_available(
    ) else torch.device("cpu")  # use cuda if available
)


loss_dir = "loss_values"
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
# Get the current timestamp to use in the filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
loss_filepath = os.path.join(loss_dir, f"loss_values_{timestamp}.csv")
model.train()
optimizer = Lion(
    model.parameters(), lr=6e-5, weight_decay=0.0001)

with open(loss_filepath, 'a', newline='') as loss_file:
    loss_writer = csv.writer(loss_file)

    for e in range(10):
        i = 0

        for epoch in range(epochs):
            optimizer.defaults["lr"]
            optimizer.zero_grad()
            tq_ = tqdm(dataloader_)  # 勾配累積なし
            optimizer.zero_grad()
            for data in tq_:
                try:
                    if i < 200:
                        lr = 6e-5*(0.005*i)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 6e-5
                    inputs = data["input_ids"].cuda()
                    inputs2 = data["input_ids2"].cuda()
                    logits = model(inputs)
                    loss = loss_fn(logits.transpose(1, 2), inputs2)
                    torch.cuda.empty_cache()
                    loss.backward()
                    tq_.set_postfix(
                        {"loss": loss.item(), "len": inputs[0].size()})
                    loss_writer.writerow([loss.item()])
                    loss_file.flush()
                    os.fsync(loss_file.fileno())
                    torch.cuda.empty_cache()
                    optimizer.step()  # スケーラーと一緒にオプティマイザのstepを実行
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                except:
                    print("err")
            #      except:
            #         print("err")
                if i % 128 == 0:
                    print()
                    str = tokenizer.decode(
                        inputs[0], skip_special_tokens=True)
                    tokens = tokenizer.encode(str)
                    _t = tokens[:len(tokens)//3]
                    _t = tokenizer.decode(_t)
                    print(str)
                    str = str[:len(str)//3]
                    print(
                        _t)
                    with torch.no_grad():
                        s = generator.generate(
                            prompt=_t,     output_len=64,     k=20,     temperature=0.4, )
                    print("\n")
                if i % 10000 == 0:
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, f"model_epoch_{e}_0_{epoch}_{i}.pth"))
                i = i+1

#
#        for epoch in range(epochs):
   #         i = 0
   #         tq = tqdm(dataloader)### 勾配累積なし
#
   #         optimizer.zero_grad()
   #         for data in tq:
   #             try:
   #                 inputs = data["input_ids"].cuda()
   #                 inputs2 = data["input_ids2"].cuda()
   #                 logits = model(inputs)
   #                 loss = loss_fn(logits.transpose(1, 2), inputs2)
   #                 torch.cuda.empty_cache()
   #                 loss.backward()
   #                 tq.set_postfix({"loss": loss.item(),"len":inputs[0].size()})
   #                 loss_writer.writerow([loss.item()])
   #                 loss_file.flush()
   #                 os.fsync(loss_file.fileno())
   #                 torch.cuda.empty_cache()
#
   #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   #                 if i%8==0:
   #                     optimizer.step()
   #                     optimizer.zero_grad()
   #                     torch.cuda.empty_cache()
   #                     for param in model.parameters():
   #                         param.data = torch.clamp(param.data, -10, 10)
#
   #             except:
   #                 print("err")
#
#
   #      #       except:
   #       #          print("err")
   #             if i % 128 == 0:
   #                 print()
   #                 str=tokenizer.decode(inputs[0],skip_special_tokens=True)
#
   #                 str=str[:len(str)//2]
   #                 print(
   #                     str)
   #                 with torch.no_grad():
   #                     s = generator.generate(
   #                         prompt=str,     output_len=100,     k=20,     temperature=1.0, )
#
   #                 print("\n")
   #             if i % 10000 == 0:
   #                 torch.save(model.state_dict(), os.path.join(
   #                     model_dir, f"model_epoch_{e}_1_{epoch}_{i}.pth"))
   #             i = i+1
#
