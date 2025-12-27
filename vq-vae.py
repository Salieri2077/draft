import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # 让残差分支在通道不一致时也能相加（你当前用法一般一致，但这样更稳）
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.proj(x)              # (B, out_channels, H, W)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + residual


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings  # K
        self.embedding_dim = embedding_dim    # D
        self.beta = beta

        # nn.Embedding 的顺序是 (num_embeddings=K, embedding_dim=D)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        # 可选：方便调试 codebook 使用情况
        self.last_encoding_inds = None

    def forward(self, latents):
        # latents: (B, D, H, W)
        B, D, H, W = latents.shape
        assert D == self.embedding_dim, f"latents channel {D} != embedding_dim {self.embedding_dim}"

        latents = latents.permute(0, 2, 3, 1).contiguous()          # (B, H, W, D)
        flat_latents = latents.view(-1, self.embedding_dim)         # (BHW, D)

        # dist: (BHW, K)
        dist = (torch.sum(flat_latents ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_latents, self.embedding.weight.t()))

        encoding_inds = torch.argmin(dist, dim=1)                   # (BHW,)
        self.last_encoding_inds = encoding_inds.view(B, H, W)        # 方便后面统计

        # quantized: (B, H, W, D)
        quantized = self.embedding(encoding_inds).view(B, H, W, D)

        # VQ Loss（标准写法：embedding_loss + beta * commitment_loss）
        commitment_loss = F.mse_loss(quantized.detach(), latents)
        embedding_loss = F.mse_loss(quantized, latents.detach())
        vq_loss = embedding_loss + self.beta * commitment_loss

        # straight-through estimator
        quantized = latents + (quantized - latents).detach()

        # back to (B, D, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, vq_loss


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dims: List[int] = None,
        beta: float = 0.25,
        img_size: int = 64,
        **kwargs
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.beta = beta
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        if hidden_dims is None:
            # 先用更容易对称的两层，下采样 2 次：64->32->16（建议你先跑通再加深）
            hidden_dims = [128, 256]

        # -------- Encoder --------
        enc_modules = []
        cur_c = in_channels
        for h_dim in hidden_dims:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv2d(cur_c, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            cur_c = h_dim

        # residual blocks（通道不变）
        for _ in range(2):  # 先用 2 个更容易调试；你想要 6 个再加回去
            enc_modules.append(ResidualLayer(cur_c, cur_c))

        # 关键：映射到 embedding_dim，使得 VQ 输入通道 = D
        enc_modules.append(nn.Sequential(
            nn.Conv2d(cur_c, embedding_dim, kernel_size=1, stride=1),
            nn.LeakyReLU()
        ))
        self.encoder = nn.Sequential(*enc_modules)

        # -------- VQ Layer --------
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta=beta)

        # -------- Decoder --------
        dec_modules = []

        dec_dims = hidden_dims[::-1]  # 不要 in-place reverse，避免影响别处逻辑
        dec_modules.append(nn.Sequential(
            nn.Conv2d(embedding_dim, dec_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        ))

        for _ in range(2):
            dec_modules.append(ResidualLayer(dec_dims[0], dec_dims[0]))

        for i in range(len(dec_dims) - 1):
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        dec_dims[i], dec_dims[i + 1],
                        kernel_size=4, stride=2, padding=1
                    ),
                    nn.LeakyReLU()
                )
            )

        # 最后恢复到图像通道数
        dec_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(dec_dims[-1], in_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*dec_modules)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, vq_loss = self.vq_layer(z_e)
        recon = self.decode(z_q)
        return recon, x, vq_loss  # 这样 loss_function 更统一

    def loss_function(self, recon_x, x, vq_loss):
        recon_loss = F.mse_loss(recon_x, x)
        loss = recon_loss + vq_loss
        return loss, recon_loss, vq_loss


if __name__ == "__main__":
    # Residual test
    x = torch.randn(4, 128, 32, 32)
    res = ResidualLayer(128, 128)
    y = res(x)
    print("Residual:", y.shape)

    # VQ test
    latents = torch.randn(4, 64, 16, 16)
    vq = VectorQuantizer(512, 64)
    q, l = vq(latents)
    print("VQ:", q.shape, "vq_loss:", l.item())

    # VQVAE test
    model = VQVAE(in_channels=3, embedding_dim=64, num_embeddings=512)
    x = torch.randn(2, 3, 64, 64)
    recon, x_in, vq_loss = model(x)
    loss, rloss, vqloss = model.loss_function(recon, x_in, vq_loss)
    print("VQVAE:", recon.shape, "loss:", loss.item())
    print("Reconstruction Loss:", rloss.item(), "VQ Loss:", vqloss.item())
    