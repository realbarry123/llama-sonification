import torch
from scipy.io import wavfile

from data import normalize, pca_reduce
from masker import Masker


class Sonifier():

    def __init__(self, shape, note_length, fs=44100):
        self._INPUT_SHAPE = shape
        self._FS = fs
        self._NOTE_LENGTH = note_length
        self._SAMPLES_PER_NOTE = int(self._FS * self._NOTE_LENGTH)

        self.config = {
            "sonification_type": "freq",
            "freq_lower": 20,
            "freq_upper": 20000,
            "do_abs": True,

            "do_interpolate": False,
            "do_diff": False,
            "pca": None,

            "channel_names": ("-b", "-f"),

            "freq_map": None
        }

        self.masker =  Masker(
            shape=(
                int(shape[0] * shape[1] * self._SAMPLES_PER_NOTE), 
                shape[2]
            ),
            pass_size=(shape[1] * self._SAMPLES_PER_NOTE)
        )


    def interpolate(self, x, scale_factor):
        """ 
        Interpolate a (time, voices)-tensor to shape (time * scale_factor, voices) 
        """
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

    
    def generate_phase(self, frequencies):
        """
        Given a (timesteps, voices)-tensor, output a (time, voices)-tensor
        of phases where time = timesteps * fs
        """
        delta = 2 * torch.pi * frequencies / float(self._FS)
        initial_phase = torch.rand(frequencies.shape[1]) * 2 * torch.pi
        phase = initial_phase.unsqueeze(0) + torch.cumsum(delta, axis=0)  # start from random phase
        phase = torch.fmod(phase, 2 * torch.pi) # wrap phase to prevent overflow
        return phase


    def mix(self, audio):
        """ 
        Sum the voice channels of a (channels, time, voices)-tensor, 
        normalize to int16 
        """
        mix = torch.sum(audio, dim=2)
        mix /= torch.max(torch.abs(mix))
        return (mix * 32767).to(torch.int16)


    def get_diff_mask(self, states: torch.Tensor):
        """
        Generate a (time, voices) mask to weigh an audio tensor's amplitude
        depending on the interpolated derivative of time
        """
        T, V = states.shape

        diff = torch.abs(torch.diff(states, dim=0))

        diff = torch.cat((
            torch.zeros(1, V), 
            diff, 
            torch.zeros(1, V)
        ), dim=0) # padded first derivative

        diff = self.interpolate(diff, self._SAMPLES_PER_NOTE)
        diff = diff[self._SAMPLES_PER_NOTE // 2 : -self._SAMPLES_PER_NOTE // 2] # crop to fit
        diff = normalize(diff, 0, 1)
        return diff


    def freq_son(self, states: torch.Tensor):
        """
        Sonify a (seq_length, layers, voices)-tensor
        """
        if list(states.shape) != list(self._INPUT_SHAPE):
            raise ValueError(f"input shape {list(states.shape)} does not match with standard shape {list(self._INPUT_SHAPE)}")
        S, L, V = self._INPUT_SHAPE
        states = states.reshape(S * L, V).float() # (time, voices)

        if self.config["pca"] is not None: 
            states = pca_reduce(states, q=self.config["pca"])

        if self.config["do_abs"]:
            states = torch.abs(states)
        
        states = normalize(states, self.config["freq_lower"], self.config["freq_upper"])

        if self.config["do_abs"]:
            states = states.clamp(min=self.config["freq_lower"])

        if self.config["do_interpolate"]:
            freq_samples = self.interpolate(states, self._SAMPLES_PER_NOTE)
        else:
            freq_samples = torch.repeat_interleave(states, self._SAMPLES_PER_NOTE, dim=0)

        phase = self.generate_phase(freq_samples)
        audio = torch.sin(phase)

        if self.config["do_diff"]:
            diff_mask = self.get_diff_mask(states, self._NOTE_LENGTH)
            audio *= diff_mask

        stereo = self.masker(audio, self.config["channel_names"])
        stereo = self.mix(stereo)

        stereo = stereo.permute(1, 0) # stupid but I have to do this

        return stereo
    

    def gain_son(self, states: torch.Tensor, freq_map: torch.Tensor):

        S, L, V = self._INPUT_SHAPE
        states = states.reshape(S * L, V).float() # (time, voices)
        if self.config["do_abs"]:
            gains = torch.abs(gains)
        gains = normalize(states, lower=0, upper=0.5)

        if len(freq_map.shape) == 1:

            if freq_map.shape[0] != gains.shape[1]:
                raise ValueError("1D freq_map must match voice dimension of gains")
            freq_map = freq_map.unsqueeze(0)
            freq_map = freq_map.repeat(S*L, 1)

        elif list(freq_map.shape) != list(gains.shape): 
            raise ValueError("2D freq_map must match shape of gains")
        
        freq_samples = torch.repeat_interleave(freq_map, self._SAMPLES_PER_NOTE, dim=0)

        if self.config["do_interpolate"]:
            gain_samples = self.interpolate(states, self._SAMPLES_PER_NOTE)
        else:
            gain_samples = torch.repeat_interleave(states, self._SAMPLES_PER_NOTE, dim=0)

        phase = self.generate_phase(freq_samples)
        audio = torch.sin(phase)
        audio *= gain_samples

        stereo = self.masker(audio, self.config["channel_names"])
        stereo = self.mix(stereo)

        stereo = stereo.permute(1, 0) # stupid but I have to do this

        return stereo
    

    def __call__(self, states: torch.Tensor, freq_map: torch.Tensor=None):
        if self.config["sonification_type"] == "freq":
            return self.freq_son(states)
        
        elif self.config["sonification_type"] == "gain":
            
            if self.config["freq_map"] == None:
                raise ValueError("config[\"freq_map\"] must be defined for gain sonification")
            
            return self.gain_son(states, self.config["freq_map"])
        
        else:
            raise ValueError(f"sonification type \"{self.config["sonification_type"]}\" not defined")
    

if __name__ == "__main__":
    states = torch.randn((3, 17, 2048))
    sonify = Sonifier(states.shape)
    wav = sonify(states).numpy()
    wavfile.write("stereo.wav", 44100, wav)