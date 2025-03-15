from typing import Union, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm.auto import tqdm

@dataclass
class ModelConfig:

    input_size_z: int = 100
    input_size_condition: int = 100
    output_size: int = 784
    d_in: int = 784
    d_label: int = 10
    hidden_size: int = 256
    d_out: int = 1
    layers: int = 2
    leaky: float = 0.2
    dropout: float = 0.5
    device: str = 'cuda'    

    #Hyperparameters
    DISC_LR:float = 0.01
    GEN_LR: float = 0.1
    MIN_LR: float = 0.000001
    DECAY_FACTOR: float = 1.00004
    DROPOUT: float = 0.5
    INIT_MOMENTUM: float = 0.5
    MAX_MOMENTUM : float = 0.7
    BATCH_SIZE : int = 50
    EPOCHS :int = 100

    WEIGHT_DECAY : float = 0.0001
    WEIGHT_DECAY_DISC : float = 0.01

    DISC_THRESHOLD : float = 0.6

class Generator(nn.Module):
    def __init__(self, input_size_z = 100, input_size_condition= 100, hidden_size = 256, output_size = 784, layers = 1,leaky = 0.2, device = 'cuda'):
        super().__init__()

        self.device = device

        self.init_layer = nn.Sequential(
            nn.Linear(input_size_z +input_size_condition, 800, device = self.device),
            nn.LeakyReLU(leaky),
        )

        self.combine = nn.Sequential(
            nn.Linear(800, hidden_size, device = self.device),
            nn.LeakyReLU(leaky),
        )

        self.layer = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, hidden_size, device = self.device),
            nn.LeakyReLU(leaky),
        ) for _ in range(layers-1)])

        self.final = nn.Sequential(
            nn.Linear(hidden_size, output_size, device = self.device),
            nn.Tanh()
        )

    def forward(self, z, y):
        """
        z: the vector 
        y: the label for the vector
        """
        z = z.to(self.device)
        y = y.to(self.device)

        combined = self.init_layer(torch.cat((z, y), dim=1))
        combined = self.combine(combined)

        for layer in self.layer:
            combined = layer(combined)
            
        return self.final(combined) # logits 

class Discriminator(nn.Module):
    def __init__(self, d_in =784, d_label = 10, hidden_size=256, d_out =1, leaky = 0.2, dropout = 0.5, device='cuda'):
        super().__init__()

        self.device = device

        self.dropout = nn.Dropout(dropout)

        self.label_embed = nn.Linear(d_label, hidden_size, device = self.device)

        self.model = nn.Sequential(
            nn.Linear(d_in + hidden_size, hidden_size, device = self.device),
            nn.LeakyReLU(leaky),
            nn.Linear(hidden_size, hidden_size, device = self.device),
            nn.LeakyReLU(leaky),
            nn.Linear(hidden_size, d_out, device = self.device)
        )

        
    def forward(self, x, y):
        x = x.to(self.device)

        y = y.to(self.device, dtype=torch.float32)
        
        y = self.label_embed(y)

        combined = torch.cat((x,y), dim =1)

        logits = self.dropout(self.model(combined))

        return logits # torch.sigmoid(combined) or torch.tanh(combined) - used with loss

class CGAN(nn.Module):
    def __init__(self, Config: ModelConfig, train_loader, test_loader):
        super().__init__()
        self.config = Config

        self.generator = Generator(self.config.input_size_z, self.config.input_size_condition, self.config.hidden_size, self.config.output_size, self.config.layers, self.config.leaky, self.config.device)
        self.discriminator = Discriminator(self.config.d_in, self.config.d_label, self.config.hidden_size, self.config.d_out, self.config.leaky, self.config.dropout, self.config.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optim_gen =nn.optim.SGD(self.generator.parameters(), lr=self.config.GEN_LR, momentum=self.config.INIT_MOMENTUM, weight_decay=self.config.WEIGHT_DECAY)
        self.optim_disc = nn.optim.SGD(self.discriminator.parameters(), lr=self.config.DISC_LR, momentum=self.config.INIT_MOMENTUM,weight_decay=self.config.WEIGHT_DECAY_DISC)

        self.scheduler_gen = nn.optim.lr_scheduler.ExponentialLR(self.optim_gen, self.config.DECAY_FACTOR) 
        self.scheduler_disc = nn.optim.lr_scheduler.ExponentialLR(self.optim_disc, self.config.DECAY_FACTOR)

        self.criterion_gen = nn.BCEWithLogitsLoss() # we use the raw logits
        self.criterion_disc = nn.BCEWithLogitsLoss() # we use the raw logits

    def train(self, accomodate_test : Optional[bool] = False):
        for epoch in tqdm(range(self.config.EPOCHS)):
            self.generator.train()
            self.discriminator.train()

            loss_disc = 0
            loss_gen = 0

            for x, y in self.train_loader:
                x = x.view(x.shape[0], -1).to(self.config.device)
                y = y.unsqueeze(1).to(self.config.device)

                self.optim_gen.zero_grad()
                self.optim_disc.zero_grad()
                
                z = torch.randn(x.shape[0], self.config.input_size_z).to(self.config.device)
                gen_out = self.generator(z, y)

                # Discriminator predictions
                disc_out_real = self.discriminator(x, y)  # D(real)
                disc_out_fake = self.discriminator(gen_out.detach(), y)  # D(fake), detach G to avoid gradient flow to Generator

                # Create real and fake labels
                real_labels = torch.ones_like(disc_out_real)
                fake_labels = torch.zeros_like(disc_out_fake)

                # Compute Discriminator loss
                loss_disc_real = self.criterion_disc(disc_out_real, real_labels)  # D(x) should be 1
                loss_disc_fake = self.criterion_disc(disc_out_fake, fake_labels)  # D(G(z)) should be 0
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

                loss_disc.backward()

                #thresholding the loss
                if loss_disc > self.config.DISC_THRESHOLD:
                    self.optim_disc.step()
        
                # Train Generator (G)
                gen_out = self.generator(z, y)
                disc_out_fake = self.generator(gen_out, y)  # D(G(z)), should be classified as real

                # Compute Generator loss
                real_labels = torch.ones_like(disc_out_fake)  # Generator wants D to classify as real
                loss_gen = self.criterion_gen(disc_out_fake, real_labels)

                loss_gen.backward()
                self.optim_gen.step()
            
                self.scheduler_gen.step()
                self.scheduler_disc.step()
            if accomodate_test:
                sample_gen, disc_loss, gen_loss = self.test()
                if (epoch+1) % 10 == 0:
                    print(f"Epoch: [{epoch+1}/{self.config.EPOCHS}], Discriminator loss: {disc_loss}, Generator loss: {gen_loss}") 
            else:
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.config.EPOCHS}], Generator Loss: {loss_gen.item():.4f}, Discriminator Loss: {loss_disc.item():.4f}')

    def test(self):
        self.generator.eval()
        self.discriminator.eval()
        with torch.inference_mode():
            sample_gen = None
            disc_loss = 0 
            gen_loss = 0
            for x, y in self.test_loader:
                x = x.view(x.shape[0], -1).to(self.config.device)
                y = y.unsqueeze(1).to(self.config.device)

                z = torch.randn(x.shape[0], self.config.input_size_z).to(self.config.device)
                gen_out = self.generator(z, y)

                disc_out_real = self.discriminator(x, y)
                disc_out_fake = self.discriminator(gen_out, y)

                real_labels = torch.ones_like(disc_out_real)
                fake_labels = torch.zeros_like(disc_out_fake)

                loss_disc_real = self.criterion_disc(disc_out_real, real_labels)
                loss_disc_fake = self.criterion_disc(disc_out_fake, fake_labels)
                loss_disc = (loss_disc_real + loss_disc_fake) / 2

                loss_gen = self.criterion_gen(disc_out_fake, real_labels)

                disc_loss += loss_disc.item()
                gen_loss += loss_gen.item()

                
                sample_gen  = torch.stack([x, gen_out], dim=1)
            
            return sample_gen, disc_loss/len(self.test_loader), gen_loss/len(self.test_loader)

        return sample_gen, disc_loss/len(test_loader), gen_loss/len(test_loader)


if __name__ == "__main__":
    config = ModelConfig()
    model = CGAN(config, train_loader, test_loader)
    print(model)
