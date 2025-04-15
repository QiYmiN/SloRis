import jax
import jax.numpy as jnp
from jax import lax
from math import exp

def gaussian(window_size, sigma):
    gauss = jnp.array([jnp.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).reshape(1, -1) 
    _2D_window = jnp.matmul(_1D_window.T, _1D_window)
    window = jnp.broadcast_to(_2D_window, (1, channel, window_size, window_size))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True, stride=None):

    mu1 = lax.conv(img1, window, (stride, stride), 'SAME')
    mu2 = lax.conv(img2, window, (stride, stride), 'SAME')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = lax.conv(img1**2, window, (stride, stride), 'SAME') - mu1_sq
    sigma2_sq = lax.conv(img2**2, window, (stride, stride), 'SAME') - mu2_sq
    sigma12 = lax.conv(img1*img2, window, (stride, stride), 'SAME') - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return jnp.mean(ssim_map)
    else:
        return jnp.mean(jnp.mean(jnp.mean(ssim_map, axis=-1), axis=-1), axis=-1)

class SSIM(object):
    def __init__(self, window_size = 4, size_average = True, stride=4):
        self.window_size = window_size
        self.size_average = size_average
        self.stride = stride

    def __call__(self, img1, img2):
        channel = img1.shape[1]
        window = create_window(self.window_size, channel)
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)










