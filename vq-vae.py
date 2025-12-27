import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(ResidualLayer,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.LeakyReLU=nn.LeakyReLU()
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        residual = x # x shape: (B,C,H,W)
        out = self.conv1(x)
        out = self.LeakyReLU(out)
        out = self.conv2(out)
        out += residual
        return out


class VectorQuantizer(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,commitment_cost=0.25,eps=1e-10):
        super(VectorQuantizer,self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.eps = eps

        # Embedding table where rows are embedding vectors.
        self.embedding = nn.Embedding(num_embeddings,embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings,1/self.num_embeddings)

    def forward(self,x):
        # x shape: (B,C,H,W)
        input_shape = x.shape
        # Move channels to the end to make flattening easier.
        x_perm = x.permute(0,2,3,1).contiguous() # (B,H,W,C)
        flat_x = x_perm.view(-1,self.embedding_dim) # (B*H*W,C)

        # Compute L2 distance between encoder outputs and embedding weights.
        distances = (torch.sum(flat_x**2,dim=1,keepdim=True)
                     + torch.sum(self.embedding.weight**2,dim=1)
                     - 2*torch.matmul(flat_x,self.embedding.weight.t())) # (B*H*W,N)

        encoding_indices = torch.argmin(distances,dim=1)
        encodings = F.one_hot(encoding_indices,self.num_embeddings).type(flat_x.dtype)

        quantized = torch.matmul(encodings,self.embedding.weight) # (B*H*W,C)
        quantized = quantized.view(input_shape[0],input_shape[2],input_shape[3],self.embedding_dim)
        quantized = quantized.permute(0,3,1,2).contiguous() # (B,C,H,W)

        # Straight-through estimator.
        quantized_st = x + (quantized - x).detach()

        # Loss terms.
        e_latent_loss = F.mse_loss(quantized.detach(),x)
        q_latent_loss = F.mse_loss(quantized,x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        avg_probs = torch.mean(encodings,dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        # Reshape indices for convenience.
        encoding_indices = encoding_indices.view(input_shape[0],input_shape[2],input_shape[3])

        return quantized_st, loss, perplexity, encoding_indices
if __name__ == "__main__":
    x = torch.randn(4,128,32,32)
    res = ResidualLayer(128,128)
    y = res(x)
    print(y.shape)
