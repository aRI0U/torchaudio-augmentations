import torch


class Delay(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        volume_factor=0.5
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.volume_factor = volume_factor

    def calc_offset(self, ms):
        return int(ms * (self.sample_rate / 1000))

    def forward(self, audio, ms):
        offset = self.calc_offset(ms)
        beginning = torch.zeros(audio.shape[0], offset).to(audio.device)
        end = audio[:, :-offset]
        delayed_signal = torch.cat((beginning, end), dim=1)
        delayed_signal = delayed_signal * self.volume_factor
        audio = (audio + delayed_signal) / 2
        return audio
