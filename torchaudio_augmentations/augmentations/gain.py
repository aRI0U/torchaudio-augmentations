import torch
from torchaudio.transforms import Vol


class Gain(torch.nn.Module):
    def forward(self, audio: torch.Tensor, gain: float) -> torch.Tensor:
        audio = Vol(gain, gain_type="db")(audio)
        return audio
