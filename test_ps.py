import time
import torch
import torchaudio

from torchaudio_augmentations.batch_augmentations.pitch_shift import BatchRandomPitchShift

wave1, sr = torchaudio.load("example.wav")
wave2, _ = torchaudio.load("example2.wav")
waveform = torch.cat((wave1, wave2)).cuda()
print(waveform.size())

n_fft = 512

ps = BatchRandomPitchShift(-5, 5, sr, n_fft=n_fft, p=1).to(waveform.device)

steps = torch.tensor([-4, 4], dtype=torch.long, device=waveform.device)
t1 = time.time()
shifted = ps(waveform, n_steps=steps)
t2 = time.time()
print(t2 - t1)

torchaudio.save("out1.wav", shifted[0].unsqueeze(0).cpu(), sr)
torchaudio.save("out2.wav", shifted[1].unsqueeze(0).cpu(), sr)
