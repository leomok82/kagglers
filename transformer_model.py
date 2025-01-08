import torch
import torch.nn as nn
# from mamba_ssm import  Mamba
from utils import *
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads = 8, bias = True):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        # Initialize dimensions
        self.embed_size = embed_size # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.head_dim = embed_size // num_heads # Dimension of each head's key, query, and value
        # scale from attention is all you need, to prevent too small gradients
        self.scale = self.head_dim ** -0.5

        # Linear layers for transforming inputs
        self.queries = nn.Linear(embed_size, embed_size, bias = bias) # Query transformation
        self.keys = nn.Linear(embed_size, embed_size, bias = bias) # Key transformation
        self.values = nn.Linear(embed_size,embed_size, bias = bias) # Value transformation
        
        self.fc_out = nn.Linear(embed_size, embed_size) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # matmul for multi dimension attention score; output is same shape as Q
        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K]) # *self.scale
        energy = energy * self.scale
        
        if mask is not None:
            mask = mask[:,None,None, :]
            energy = energy.masked_fill(mask == 0, float("-1e20"))
    
        attention = torch.softmax(energy , dim = 3)
        
        # Multiply by values to obtain the final output (query and key r gone)
        output = torch.einsum("nhql,nlhd ->nqhd", [attention, V]) 
        
        return output
        
 
    def forward(self, Q, K, V, mask=None):
        N, seq_len, embedding_size = Q.size()

        # first linear transformation before split heads. Could try split heads first by changing linear size
        V= self.values(V)
        K = self.keys(K)
        Q = self.queries(Q)
        # split heads (bs, 256, 8, 64)
        ### V.size edited
        V = V.reshape(V.size(0), V.size(1), self.num_heads, self.head_dim)
        K = K.reshape(K.size(0), K.size(1), self.num_heads, self.head_dim)
        Q = Q.reshape(N, seq_len, self.num_heads, self.head_dim)

        


        # Perform scaled dot-product attention
        out = self.scaled_dot_product_attention(Q, K, V, mask).reshape(N, seq_len, self.embed_size)

        # Combine heads and apply output transformation
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module): #Feed forward block
    def __init__(self, embed_size, heads, dropout, forward_expansion ):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # Similar to batch norm, but takes average for every sample
        self.norm2 = nn.LayerNorm(embed_size)

        #feed forward
        self.feed_forward = nn.Sequential(
            #expands to larger size then returns to embed size
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)


        
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, Q, mask = None):
        Q =self.norm1(Q)
        attention = self.attention(Q, Q, Q, mask) # Q K V

        #skip connection 
        x = self.dropout(self.norm2(attention + Q))

        #feed forward 
        forward = self.feed_forward(x)
        out = self.dropout(x + forward )
        return out
    
class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout = dropout, forward_expansion = forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # shape (bs, num_patches, seqlen/patches, vocab_size - 1)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x,  mask)
        return x
    






class PositionalEncoding(nn.Module):
    def __init__(self, embed_size,num_patches):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(num_patches).unsqueeze(1)  # Shape: (max_length, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pe = torch.zeros(num_patches, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        seq_len = x.size(1)  # Get sequence length
        return self.positional_encoding[:seq_len, :] 
class RoPE(nn.Module):
    def __init__(self, embed_size, num_patches):
        super(RoPE, self).__init__()
        self.embed_size = embed_size
        self.num_patches = num_patches

        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
        sinusoidal_pos = torch.zeros(num_patches, embed_size)
        sinusoidal_pos[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_pos[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('sinusoidal_pos', sinusoidal_pos)

    def forward(self, x):
        batch_size, num_patches, num_categories = x.shape
        
        
        # Get positional encodings, trimmed to the required sequence length
        pos_encoding = self.sinusoidal_pos[:num_patches, :]
       
        # Reshape x and positional encoding to account for patches and sequence within patches
        x_reshaped = x.view(batch_size, num_patches, -1, 2)
        pos_reshaped = pos_encoding.view(num_patches, -1, 2)

        # Ensure the shapes match for element-wise operations
        assert x_reshaped.shape[-2] == pos_reshaped.shape[-2], \
            f"Shape mismatch: x_reshaped {x_reshaped.shape}, pos_reshaped {pos_reshaped.shape}"

        # Apply rotary positional encoding
        x_sin = x_reshaped[..., 0] * pos_reshaped[..., 0] - x_reshaped[..., 1] * pos_reshaped[..., 1]
        x_cos = x_reshaped[..., 0] * pos_reshaped[..., 1] + x_reshaped[..., 1] * pos_reshaped[..., 0]

        # Stack and reshape back to the original patch-wise structure
        x_rotated = torch.stack((x_sin, x_cos), dim=-1).view(batch_size, num_patches, -1)

        return x_rotated
    

    
class Transformer(nn.Module):
    def __init__(self,  embed_size = 64,  num_layers = 2, forward_expansion = 4, heads = 8, dropout = 0, max_length = 66, window_size = 30,n_features = 188
                 ):
        super(Transformer, self).__init__()
        # encoder

            
        self.max_length = max_length


        self.pe = PositionalEncoding(embed_size, max_length)


        
        
        # self.attention_pool = AttentionPooling(embed_size)
        self.embedding_layer = nn.Linear(window_size * n_features, embed_size)
   
        self.encoder = Encoder( embed_size, num_layers, heads, forward_expansion, dropout)

        
        self.final_layer = nn.Linear(embed_size  , 1) 

        

    def forward(self, x):
        

        # x shape (bs, n_symbols, window_size, n_features ) 


        x = x[:,:self.max_length]

        # Reshape to (bs, n_symbols, window_size * n_features)
        x = x.view(x.shape[0], x.shape[1], -1)

        # Embedding layer to convert to (bs, n_symbols, embed_size)
        x = self.embedding_layer(x)



        # encoder 
        x = self.encoder(x+self.pe(x), None)  #+self.pe(x)


        # Patch merging
        
        x = self.final_layer(x)

        return x
    def ladder(self, x, z, mutation_rates = [0.1,0.5,1], device ='cuda', pos_rate = 0.01):
        ladder_losses = 0
        def triplet_loss_func( dist1 ,dist2, div ):
    
            return torch.clamp(div  + dist1 - dist2 , min=1e-12)**2
        
        
        positive = noisy(x, pos_rate) 
        positive_z = self.forward(positive.to(device))
        pos_dist = self.dist_func(z, positive_z)

        for rate in mutation_rates:
            negative =noisy(x, rate)
            negative = negative.to(device)
            negative_z = self.forward(negative)
            neg_dist = self.dist_func(z, negative_z)
            ladder_losses+= triplet_loss_func( pos_dist ,neg_dist, 1 )
        
        
        ladderLoss = ladder_losses.mean()/ max(len(mutation_rates),1)
        return ladderLoss