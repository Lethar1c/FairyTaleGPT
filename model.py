import torch
from torch.nn import Linear, Embedding, TransformerEncoderLayer
import torch.nn as nn

#
# class MiniGPT(nn.Module):
#     def __init__(self, device, vocab_size, max_seq_length, embedding_dim=256, num_heads=8, dropout=0.1, batch_first=False):
#         super().__init__()
#         self.device = device
#         self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.embedding_dim = embedding_dim
#         self.position_embedding = Embedding(max_seq_length, embedding_dim)
#         self.decoder1 = TransformerEncoderLayer(embedding_dim, num_heads, 4*embedding_dim,
#                                                 dropout, device=device, batch_first=batch_first)
#         self.decoder2 = TransformerEncoderLayer(embedding_dim, num_heads, 4 * embedding_dim,
#                                                 dropout, device=device, batch_first=batch_first)
#         self.decoder3 = TransformerEncoderLayer(embedding_dim, num_heads, 4 * embedding_dim,
#                                                 dropout, device=device, batch_first=batch_first)
#         self.linear = Linear(embedding_dim, vocab_size)
#
#     def forward(self, x):
#         mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1])
#
#         x = self.embedding(x) + self.position_embedding(torch.arange(x.shape[1]).to(self.device))
#         x = self.decoder1(x, src_mask=mask, is_causal=True)
#         x = self.decoder2(x, src_mask=mask, is_causal=True)
#         x = self.decoder3(x, src_mask=mask, is_causal=True)
#         x = self.linear(x)
#         return x
#
#     @torch.no_grad()
#     def prompt_one(self, x, context_len=256, max_len=1000):
#         # 1 - BOS, 2 - EOS
#         ans = x.copy()
#         l = len(x)
#         while l < max_len and ans[-1] != 2:
#             context = ans[-context_len:] if l >= context_len else ans
#             model_x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(self.device)
#             # print(model_x.shape)
#             logits = self(model_x).squeeze()[-1]
#
#             probs = torch.softmax(
#                 logits,
#                 dim=-1
#             )
#
#             pred_token = torch.multinomial(
#                 probs,
#                 num_samples=1
#             ).item()
#
#             ans.append(pred_token)
#             l += 1
#         return ans


class MiniGPT(nn.Module):
    def __init__(self, device, vocab_size, max_seq_length, embedding_dim=384, num_heads=8, dropout=0.1, batch_first=False, num_layers=6):
        super().__init__()
        self.device = device
        self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.position_embedding = Embedding(max_seq_length, embedding_dim)
        self.decoders = nn.ModuleList(
            [TransformerEncoderLayer(embedding_dim, num_heads, 4*embedding_dim,
                                     dropout, device=device, batch_first=batch_first)
            for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear = Linear(embedding_dim, vocab_size)


    def forward(self, x):
        print(x.shape)
        x = x.to(self.device)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(self.device)

        x = self.embedding(x) + self.position_embedding(torch.arange(x.shape[1]).to(self.device))
        for decoder in self.decoders:
            x = decoder(x, src_mask=mask)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x

    @torch.no_grad()
    def prompt_one(self, x, context_len=256, max_len=1000, temperature=1.0, k=20):
        # 1 - BOS, 2 - EOS
        ans = x.copy()
        l = len(x)
        while l < max_len and ans[-1] != 2:
            context = ans[-context_len:] if l >= context_len else ans
            model_x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(self.device)
            logits = self(model_x).squeeze()[-1] / temperature

            values, indices = torch.topk(logits, k)

            probs = torch.softmax(
                values,
                dim=-1
            )

            pred_token = torch.multinomial(
                probs,
                num_samples=1
            ).item()

            ans.append(pred_token)
            l += 1
        return ans