import torch

class Masker():

    @staticmethod
    def _get_lr_masks(voices: int):
        """
        Generate L and R stereo masks of size (1, voices), with linear
        variation across the voices dimension. To be multiplied with
        audio tensor. 
        """
        l_mask = torch.arange(1, 0, -1/voices).unsqueeze(0)
        r_mask = torch.ones(voices).unsqueeze(0) - l_mask

        return l_mask, r_mask

    @staticmethod
    def _get_fb_masks(timesteps: int, pass_size: int):
        """
        Generate F and B stereo masks of size (time, 1), with linear
        variation across the layer dimension. To be multiplied with
        audio tensor.
        """
        f_mask = torch.arange(1, 0, -1/pass_size)
        b_mask = torch.ones(pass_size) - f_mask

        n_passes = int(timesteps / pass_size)
        print(n_passes * pass_size, timesteps)
        f_mask = f_mask.repeat((n_passes,)).unsqueeze(1)
        b_mask = b_mask.repeat((n_passes,)).unsqueeze(1)

        return f_mask, b_mask
    
    def __init__(self, shape, pass_size):
        T, V = shape
        self.l, self.r = self._get_lr_masks(V)
        self.f, self.b = self._get_fb_masks(T, pass_size)
    
    def _mask(self, x, channel_name):
        """
        Mask a (time, voices)-tensor x based on channel name
        """
        x = x.clone()

        if channel_name[0] == "l":
            x *= self.l
        elif channel_name[0] == "r":
            x *= self.r
        elif channel_name[0] != "-":
            raise ValueError(f"channel name \"{channel_name}\" not recognized")
        
        if channel_name[1] == "f":
            x *= self.f
        elif channel_name[1] == "b":
            x *= self.b
        elif channel_name[1] != "-":
            raise ValueError(f"channel name \"{channel_name}\" not recognized")
        
        return x
    
    def __call__(self, x, channel_names: tuple):
        """
        Create a stereo tensor of size (channels, time, voices) from a 
        mono tensor of size (time, voices) and a list of channel names
        """
        T, V = x.shape
        stereo = torch.zeros(len(channel_names), T, V)

        for i in range(len(channel_names)):
            stereo[i] = self._mask(x, channel_names[i])

        return stereo