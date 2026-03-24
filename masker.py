import torch

class Masker():

    @staticmethod
    def _get_lr_masks(voices: int):
        """
        Generate L and R stereo masks of size (1, voices), with 
        linear variation across the voices dimension. To be multiplied
        with audio tensor. 
        """
        l_mask = torch.arange(1, 0, -1/voices).unsqueeze(0)
        r_mask = torch.ones(voices).unsqueeze(0) - l_mask

        return l_mask, r_mask

    @staticmethod
    def _get_fb_masks(timesteps: int, n_layers: int):
        """
        Generate F and B stereo masks of size (timesteps, 1), with
        linear variation across the layer dimension. To be multiplied
        with audio tensor.
        """
        f_mask = torch.arange(1, 0, -1/n_layers)
        b_mask = torch.ones(n_layers) - f_mask

        n_passes = int(timesteps / n_layers)
        f_mask = f_mask.repeat((n_passes,)).unsqueeze(1)
        b_mask = b_mask.repeat((n_passes,)).unsqueeze(1)

        return f_mask, b_mask
    
    def __init__(self, shape, n_layers=17):
        T, V = shape
        self.l, self.r = self._get_lr_masks(V)
        self.f, self.b = self._get_fb_masks(T, n_layers)
    
    def _mask(self, x, channel_name):
        """
        Mask a tensor x of size (T, V) based on channel name
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
        Create a stereo tensor of size (C, T, V) from a mono tensor of size (T, V) 
        and a list of channel names
        """
        T, V = x.shape
        stereo = torch.zeros(len(channel_names), T, V)

        for i in range(len(channel_names)):
            stereo[i] = self._mask(x, channel_names[i])

        return stereo