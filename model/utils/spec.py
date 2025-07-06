import torch
import torch.nn as nn

import math
from typing import Optional, Literal

class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 80,
        window_fn: Literal['hann', 'hamming'] = "hann",
        power: float = 2.0,
        frame_norm: bool = False,
        window_norm: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        slaney_norm: bool = False,
        mel_scale: Literal['htk', 'slaney'] = "htk",
    ) -> None:
        super().__init__()
        self.hop_length = hop_length

        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            frame_norm=frame_norm,
            window_norm=window_norm,
            center=center,
            pad_mode=pad_mode,
            slaney_norm=slaney_norm,
            mel_scale=mel_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(
            torch.clamp_min(self.melspectrogram(x), min=1e-5)
        )

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 80,
        window_fn: Literal['hann', 'hamming'] = "hann",
        power: float = 2.0,
        frame_norm: bool = False,
        window_norm: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        slaney_norm: bool = False,
        mel_scale: Literal['htk', 'slaney'] = "htk"
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.frame_norm = frame_norm
        self.window_norm = window_norm
        self.n_mels = n_mels
        self.f_max = f_max
        self.f_min = f_min
        
        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            frame_norm=self.frame_norm,
            window_norm=self.window_norm,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, slaney_norm, mel_scale
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel_scale(self.spectrogram(waveform))
    
class Spectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: str = "hann",
        power: Optional[float] = 2.0,
        frame_norm: bool = False,
        window_norm: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True
    ) -> None:
        super().__init__()
        assert window_fn in ["hann", "hamming"]

        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.register_buffer("window", torch.hann_window(self.win_length) if window_fn == "hann" else torch.hamming_window(self.win_length))
        self.pad = pad
        self.power = power
        self.frame_norm = frame_norm
        self.window_norm = window_norm
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            waveform = torch.nn.functional.pad(waveform, (self.pad, self.pad), "constant")

        inputBatch = False
        if waveform.ndim == 2:
            batch_size, signal_length = waveform.size()
            inputBatch = True
        else:
            batch_size = 1
            signal_length = waveform.size(0)

        spec_f = torch.stft(
            input=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.frame_norm,
            onesided=self.onesided,
            return_complex=True,
        )

        if inputBatch:
            spec_shape = [batch_size, self.n_fft//2 + 1, signal_length//self.hop_length + 1]
        else:
            spec_shape = [self.n_fft//2 + 1, signal_length//self.hop_length + 1]

        spec_f = spec_f.view(spec_shape)

        if self.window_norm:
            spec_f /= self.window.pow(2.0).sum().sqrt()
        
        if self.power is not None:
            if self.power == 1.0:
                return spec_f.abs()
            return spec_f.abs().pow(self.power)
        
        return spec_f
    
class MelScale(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        slaney_norm: bool = False,
        mel_scale: Literal['htk', 'slaney'] = "htk",
    ) -> None:
        super().__init__()
        self.n_stft = n_stft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.slaney_norm = slaney_norm
        self.mel_scale = mel_scale
        
        self.register_buffer("fb", self.melscale_fbanks())

        self.__fmin = 0.0
        self.__f_sp = 200.0 / 3
        self.__logsteps = math.log(6.4) / 27.0
        self.__min_log_hz = 1000.0
        self.__min_log_mel = (self.__min_log_hz - self.__fmin) / self.__f_sp

    def _hz_to_mel(self, freq: float) -> float:
        if self.mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + (freq / 700.0))

        # Fill in the linear part
        mels = (freq - self.__fmin) / self.__f_sp

        # Fill in the log-scale part
        if freq >= self.__min_log_hz:
            mels = self.__min_log_mel + math.log(freq / self.__min_log_hz) / self.__logsteps

        return mels

    def _mel_to_hz(self, mels: torch.Tensor) -> torch.Tensor:
        if self.mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # Fill in the linear scale
        freqs = self.__fmin + self.__f_sp * mels

        # And now the nonlinear scale
        log_t = mels >= self.__min_log_mel
        freqs[log_t] = self.__min_log_hz * torch.exp(self.__logsteps * (mels[log_t] - self.__min_log_mel))

        return freqs

    def _create_triangular_filterbank(
        self,
        all_freqs: torch.Tensor,
        f_pts: torch.Tensor,
    ) -> torch.Tensor:
        f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
        # create overlapping triangles
        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)

        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))

        return fb

    def melscale_fbanks(self) -> torch.Tensor:
        # freq bins
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # calculate mel freq bins
        m_min = self._hz_to_mel(self.f_min)
        m_max = self._hz_to_mel(self.f_max)

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = self._mel_to_hz(m_pts)

        # create filterbank
        fb = self._create_triangular_filterbank(all_freqs, f_pts)

        if self.slaney_norm:
            enorm = 2.0 / (f_pts[2 : self.n_mels + 2] - f_pts[:self.n_mels])
            fb *= enorm.unsqueeze(0)

        return fb

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return mel_specgram