from config import ModelArgs
import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalization(nn.Module):
    def __init__(
            self,
            embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        self.rmsnorm_layer = torch.nn.RMSNorm(normalized_shape=embeddings_dims)

    def forward(self, x):

        x = self.rmsnorm_layer(x)
        return x
    
class RotaryEmbeddings(nn.Module):
    def __init__(
            self,
            device,
            embeddings_dims: int = ModelArgs.embeddings_dims,
            block_size: int = ModelArgs.block_size,
            batch_size: int = ModelArgs.batch_size
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.theta = 0
        self.device = device

    def apply_rope(self, seq):

        batch_size, seq_len, embeds_dims = seq.shape

        positions = torch.arange(0, embeds_dims, 2, dtype=torch.float32, device=self.device).unsqueeze(0)

        theta = 10000 ** (-2 * (positions) / embeds_dims)

        angles = positions * theta

        angles = angles.expand(seq_len, -1)

        x_reshaped = seq.view(batch_size, seq_len, embeds_dims // 2, 2)

        cos_angles = torch.cos(angles)

        sin_angles = torch.sin(angles)

        out = torch.stack([x_reshaped[..., 0]*cos_angles - 
                           (x_reshaped[...,1] * sin_angles), 
                           x_reshaped[...,1] * cos_angles + 
                           x_reshaped[..., 0] * sin_angles], 
                           dim=-1)
        
        out = out.view(batch_size, seq_len, embeds_dims)

        return out
    
    def forward(self, x):

        res = self.apply_rope(x)

        return res
    
class RotaryAttentionHead(nn.Module):

    def __init__(
            self,
            device,
            embeddings_dims: int = ModelArgs.embeddings_dims,
            no_of_heads: int = ModelArgs.no_of_heads,
            attn_dropout: int = ModelArgs.attn_dropout
    ):
        super().__init__()

        self.head_size = embeddings_dims // no_of_heads

        self.query = nn.Linear(in_features=embeddings_dims,
                               out_features=self.head_size,
                               bias=False, dtype=torch.float32,
                               device=device)
        
        self.key = nn.Linear(in_features=embeddings_dims,
                               out_features=self.head_size,
                               bias=False, dtype=torch.float32,
                               device=device)
        
        self.value = nn.Linear(in_features=embeddings_dims,
                               out_features=self.head_size,
                               bias=False, dtype=torch.float32,
                               device=device)
        
        self.rope = RotaryEmbeddings(embeddings_dims=self.head_size,  
                                     device = device)
        
        self.dropout = nn.Dropout(p = attn_dropout)
        
        self.device = device

    def forward(self, x):

        batch, block_size, embeddings_dims = x.shape

        query = self.query(x)

        key= self.key(x)

        values = self.value(x)

        rotary_q = self.rope(query)
        rotary_k = self.rope(key)

        masked = torch.tril(torch.ones((block_size, block_size), requires_grad=False, device = self.device))

        weights = rotary_q.permute(2,0,1) @ rotary_k.permute(2,0,1).transpose(-2,-1)

        weights_masked = weights.masked_fill(masked == 0, float('-inf'))
        
        scaled_weights = weights_masked / (torch.sqrt(torch.tensor(key.shape[-1])))
        
        scaled_weights = F.softmax(scaled_weights, dim=-1)
        
        value = scaled_weights @ values
        
        out = self.dropout(value)

        return out
    
class MQA(nn.Module):
    def __init__(
            self,
            device,
            no_of_q_heads: int,
            embeddings_dims: int = ModelArgs.embeddings_dims,
            block_size: int = ModelArgs.block_size
    ):
        super().__init__()

        self.no_of_kv_heads = 2
        
        self.head_size = embeddings_dims // no_of_q_heads

        self.rotary = RotaryEmbeddings(embeddings_dims=self.head_size,  device = device)

        self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  dtype=torch.float32, bias=False,  device = device)
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=self.head_size * self.no_of_kv_heads, out_features=embeddings_dims,  dtype=torch.float32, bias=False,  device = device)
        self.device = device
        self.multi_query = nn.ModuleList([nn.Linear(in_features=embeddings_dims, out_features=self.head_size,  bias=False,  device = self.device) for _ in range(self.no_of_kv_heads)])

    def scaled_dot_product(self, q, k ,v, block_size):

        q = self.rotary(q)

        masked_table = torch.tril(torch.ones((block_size, block_size),  requires_grad=False,  device = self.device))

        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
        weights_normalized = nn.functional.softmax(masked_values, dim=-1) #Normalize along the embeddings dimension for all the tokens
        weights_normalized = self.dropout(weights_normalized)
        out = weights_normalized @ v
        return out
    
    def forward(self, x):

        batch, block_size, embeddings_dims = x.shape

        key = self.key(x)
        values = self.value(x)

        rotary_key = self.rotary(key)
        multi_query_concat = torch.cat([self.scaled_dot_product(query(x), rotary_key, values, block_size) for query in self.multi_query], dim=-1)

        linear_layer = self.linear_layer(multi_query_concat)

        return linear_layer
    
class GQA(nn.Module):
    def __init__(
            self,
            device,
            embeddings_dims: int = ModelArgs.embeddings_dims,
            block_size: int = ModelArgs.block_size,
            mqa_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()

        self.no_of_q_heads = ModelArgs.no_of_heads // mqa_heads

        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)

        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_q_heads, 
                                      out_features=embeddings_dims , 
                                      dtype=torch.float32,  
                                      bias=False,  device = device)
        
        self.device = device

        self.mqa = nn.ModuleList([MQA(no_of_q_heads=self.no_of_q_heads,
                                      embeddings_dims=embeddings_dims,
                                      device=self.device,
                                      block_size=block_size) for _ in range(self.no_of_q_heads)])
        
    def forward(self, x):

        batch, block_size, embeddings_dims = x.shape

        grouped_query_concat = torch.cat([group(x) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat)

        out = self.dropout(linear_layer)

        return out
    
class Swish(nn.Module):
    def __init__(
            self,
            device,
            block_size: int = ModelArgs.block_size,
            embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        swish = x * self.sig(x)

        return swish
    
class SWiGLU(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()
        self.hidden_dims = int(2 * ( 4 * embeddings_dims) / 3)
        self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, dtype=torch.float32,  device = device)
        self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, dtype=torch.float32,  device = device)

    def forward(self, x):
        swish_res = self.swish(self.linear_layer1(x))
        x_V = self.linear_layer2(x)
        res = torch.mul(swish_res, x_V)
        out = self.linear_layer3(res)
        return out
    
class FFN(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                   dropout = ModelArgs.dropout

                 ):
        super().__init__()

        self.swiglue = SWiGLU(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):

        x = self.swiglue(x)

        x = self.dropout(x)

        return x
    
class DecoderLayer(nn.Module):
    def __init__(self,
                  device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,

                 ) :
        super().__init__()


        self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, mqa_heads=2,  device = device)
        self.norm1 = Normalization(embeddings_dims=embeddings_dims)
        self.norm2 = Normalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = x + self.gqa(self.norm1(x))
        x = x + self.feedforward_network(self.norm2(x))
        return x
    
class Llama(nn.Module):
    def __init__(self,
                device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                  dropout = ModelArgs.dropout

                 ) :
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  dtype=torch.float32,  device = device)
        self.decoder = nn.Sequential(*[DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers)])
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,  dtype=torch.float32,  device = device)
        self.dropout = nn.Dropout(p = dropout)
        
        self.embeddings.weight = self.linear_layer.weight
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                     
                    
    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        x = self.linear_layer(x)
    
        return x
