import torch

class SNRLoss_dB_Signals(torch.nn.Module):
    def __init__(self, N=1024, pulse_band=[45/60., 250/60.], Fs=30):
        super(SNRLoss_dB_Signals, self).__init__()
        self.N = N
        self.Fs = Fs
        self.pulse_band = torch.tensor(pulse_band, dtype=torch.float32)


    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, ):
        device = outputs.device
        self.pulse_band = self.pulse_band.to(device)
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()
        N_samp = outputs.shape[-1]
        wind_sz = int(self.N/256)

        f = torch.linspace(0, self.Fs/2, int(self.N/2)+1, dtype=torch.float32).to(device)

        min_idx = torch.argmin(torch.abs(f - self.pulse_band[0]))
        max_idx = torch.argmin(torch.abs(f - self.pulse_band[1]))

        outputs = outputs.view(-1, N_samp)
        targets = targets.view(-1, N_samp)

        # Generate GT heart indices from GT signals
        Y = torch.fft.rfft(targets, n=self.N, dim=1, norm='forward')
        Y2 = torch.abs(Y)**2
        HRixs = torch.argmax(Y2[:,min_idx:max_idx],axis=1)+min_idx

        X = torch.fft.rfft(outputs, n=self.N, dim=1, norm='forward')

        P1 = torch.abs(X)**2

        # Calc SNR for each batch
        losses = torch.empty((X.shape[0],), dtype=torch.float32)
        for count, ref_idx in enumerate(HRixs):
            lower_lim = max([0, ref_idx-wind_sz])
            pulse_freq_amp = torch.sum(P1[count, lower_lim:ref_idx+wind_sz])
            other_avrg = (torch.sum(P1[count, min_idx:lower_lim])+torch.sum(P1[count, ref_idx+wind_sz:max_idx]))
            losses[count] = -10*torch.log10(pulse_freq_amp/(other_avrg+1e-7))
        losses.to(device)
        return torch.mean(losses)