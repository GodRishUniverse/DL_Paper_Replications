import torch
from torch import nn
from typing import Union, List, Tuple, Optional, Dict

from einops import rearrange
from einops.layers.torch import Rearrange

from dataclasses import dataclass


from torchvision import datasets, transforms

from tqdm.auto import tqdm

@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_classes: int = 1000
    dim: int = 768
    hidden_dim: int = 3072
    batch_first_MSA: bool = True
    num_heads: int = 12
    num_layers: int = 12
    device: str = 'cuda'

    classify: bool = True
    class_dim: int = 768
    hid_class_dim: int = 3072
    # dropout: float = 0.1 - not used in this implementation


class MLP(nn.Module):
    def __init__(self, config: ModelConfig, out_dim: Optional[int] = None):
        super().__init__()
        self.out_dim = config.dim
        if out_dim is not None:
            self.out_dim = out_dim

        self.layer = nn.Sequential(
            nn.LayerNorm(config.dim, device = config.device),
            nn.Linear(config.dim, config.hidden_dim, device = config.device),
            nn.GELU(),
            nn.Linear(config.hidden_dim, self.out_dim, device= config.device),
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x
    
class MSA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.device = config.device
        self.mha = nn.MultiheadAttention(config.dim, config.num_heads, batch_first=config.batch_first_MSA, device = config.device)
    def forward(self, x):
        x = x.to(self.device)
        x, _ = self.mha(x, x, x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # need to figure out how to compute D and then get the z array from x
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MSA(config=config),
                MLP(config=config)
            ])
            for _ in range(config.num_layers)
        ])

        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(config.dim, device = config.device)
        
    def forward(self, x):
        for attn, ffn in self.layers:
            x = x + attn(self.layer_norm(x)) 
            x = x + ffn(self.layer_norm(x))
        x = self.dropout(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, config: ModelConfig):
        image_size = config.image_size
        super().__init__()
        self.device = config.device
        # image size can be be H * W
        H, W = image_size, image_size # square image
        patch_size = config.patch_size
        assert H % patch_size == 0 and W % patch_size == 0, "Height and Width must be divisible by patch size"
        # number of patches
        num_patches = (H // patch_size) * (W // patch_size)
        # patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_size * patch_size * config.num_channels, device = config.device),
            nn.Linear(patch_size * patch_size * config.num_channels, config.dim, device = config.device),
            nn.LayerNorm(config.dim, device = config.device)
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, config.dim)).to(self.device)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.dim)).to(self.device)
        self.transformer = Transformer(config=config)


        if config.classify:
            self.mlp_head = MLP(config=config, out_dim=config.class_dim)
        
    
    def forward(self, x):
        b, _, _, _ = x.shape
    
        # Create patch embeddings
        x = self.to_patch_embedding(x)
        x = x.to(self.device)
        
        # Expand class token to match batch size
        cls_tokens = self.class_token.expand(b, -1, -1)
        cls_tokens = cls_tokens.to(self.device)
        
        # Concatenate class token with patch embeddings
        z = torch.cat((cls_tokens, x), dim=1)

        z = z + self.pos_embedding
        z = self.transformer(z)
        print(z.shape, " After Transformer")
        
        if hasattr(self, 'mlp_head'):
            z = self.mlp_head(z)

            logits = z[:, 0, :] # Get the logits for the class token which is the first token
            return z, logits
        return z, None


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(root='./ViT/train_vit', train  = True, download=True, transform=img_transform)
    test_dataset = datasets.CIFAR100(root='./ViT/test_vit', train  = False, download=True, transform=img_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    config = ModelConfig(
        image_size = 32,
        num_channels = 3,
        dim = 3072,
        num_heads = 12,
        num_layers = 12,
        hidden_dim = 768,
        patch_size = 8,
        num_classes = 100,
        classify=True,
        device = device
    )
    vit = ViT(config=config)
    vit = vit.to(device)
    print(vit)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(vit.parameters(), lr=0.0001, weight_decay=0.1)
    num_epochs = 10

    for epoch in tqdm(range(num_epochs)):
        vit.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            z, logits = vit(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        with torch.inference_mode():
            vit.eval()
            correct = 0
            total = 0
            for images, labels in test_dataset:
                images = images.to(device)
                labels = labels.to(device)
                z, logits = vit(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Test Accuracy of the model on the 10000 test images: {100 * correct // total} %')
