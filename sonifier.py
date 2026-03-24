import torch
from scipy.io import wavfile

from data import normalize
from masker import Masker


class Sonifier():

    def __init__(self, shape):
        self.input_shape = shape
        self.fs = 44100
        self.configs = {
            "note_length": 0.12,
            "freq_lower": 50,
            "freq_upper": 6000,

            "do_interpolate": False,
            "do_diff": False,
        }

        samples_per_note = int(self.fs * self.configs["note_length"])
        
        self.masker =  Masker(
            shape=(
                int(shape[0] * shape[1] * samples_per_note), 
                shape[2]
            ),
            pass_size=(shape[1] * samples_per_note))
        
        self.channel_names = ("-b", "-f")


    def interpolate(self, x, scale_factor):
        """ Interpolate a (time, voices)-tensor to shape (time * scale_factor, voices) """
        T, V = x.shape
        x = x.permute(1, 0).view(1, V, T) # (1, V, T)

        x = torch.nn.functional.interpolate(
            x, 
            scale_factor=scale_factor, 
            mode='linear', 
            align_corners=False
        ) # (1, V, T * scale)

        x = x.squeeze().permute(1, 0) # (T * scale, V)
        return x


    def generate_phase(self, frequencies, fs=44100):
        """
        Given a (timesteps, voices)-tensor, output a (time, voices)-tensor of phases
        where time = timesteps * fs
        """
        delta = 2 * torch.pi * frequencies / float(fs) # phase increment per sample
        phase = torch.cumsum(delta, axis=0) # integrate
        phase = torch.fmod(phase, 2 * torch.pi) # wrap phase to prevent overflow
        return phase


    def mix(self, audio):
        """ Sum the voice channels of a (channels, time, voices)-tensor, normalize to int16 """
        mix = torch.sum(audio, dim=2)
        mix /= torch.max(torch.abs(mix))
        return (mix * 32767).to(torch.int16)


    def get_diff_mask(self, history: torch.Tensor):
        """
        Generate a (time, voices) mask to weigh an audio tensor's amplitude
        depending on the interpolated derivative of time
        """
        T, V = history.shape
        samples_per_note = int(self.fs * self.configs["note_length"])

        diff = torch.abs(torch.diff(history, dim=0))

        diff = torch.cat((
            torch.zeros(1, V), 
            diff, 
            torch.zeros(1, V)
        ), dim=0) # padded first derivative

        diff = self.interpolate(diff, samples_per_note)
        diff = diff[samples_per_note // 2 : -samples_per_note // 2] # crop to fit
        diff = normalize(diff, 0, 1)
        return diff


    def __call__(self, states: torch.Tensor):
        """
        Sonify a (seq_length, layers, voices)-tensor
        """
        if list(states.shape) != list(self.input_shape):
            raise ValueError(f"input shape {list(states.shape)} does not match with standard shape {list(self.input_shape)}")
        S, L, V = self.input_shape
        states = states.reshape(S * L, V).float() # (time, voices)
        states = normalize(states, 50, 6000)

        samples_per_note = int(self.fs * self.configs["note_length"])

        if self.configs["do_interpolate"]:
            freq_samples = self.interpolate(states, samples_per_note)
        else:
            freq_samples = torch.repeat_interleave(states, samples_per_note, dim=0)

        phase = self.generate_phase(freq_samples)
        audio = torch.sin(phase)

        if self.configs["do_diff"]:
            diff_mask = self.get_diff_mask(states, self.configs["note_length"])
            audio *= diff_mask

        stereo = self.masker(audio, self.channel_names)
        stereo = self.mix(stereo)

        stereo = stereo.permute(1, 0) # stupid but I have to do this

        return stereo
    

if __name__ == "__main__":
    states = torch.randn((3, 17, 2048))
    sonify = Sonifier(states.shape)
    wav = sonify(states).numpy()
    wavfile.write("stereo.wav", 44100, wav)