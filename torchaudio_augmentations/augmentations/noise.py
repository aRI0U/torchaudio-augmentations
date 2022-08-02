import torch


class Noise(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio, snr):
        std = torch.std(audio)
        noise_std = snr * std

        noise = noise_std * torch.randn_like(audio)

        return audio + noise
