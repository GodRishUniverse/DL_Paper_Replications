from paperreplications.ViT.script import ViT, ModelConfig

print("Testing ViT...")
config = ModelConfig(image_size=224, patch_size=16, dim=512, hidden_dim=128, num_classes=1000, num_heads=8, num_layers=6, num_channels=3, device="cpu")
vit = ViT(config)
print(vit)
