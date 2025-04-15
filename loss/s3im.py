import jax.numpy as jnp
from jax import random
from .ssim import SSIM
from skimage.metrics import structural_similarity

class S3IM(object):
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=32, patch_width=32):
        self.kernel_size = kernel_size
        self.stride = stride 
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)

    def __call__(self, src_vec, tar_vec):
        loss = 0.0
        s1,s2,s3 = jnp.split(src_vec, 3, axis=-1)
        t1,t2,t3 = jnp.split(tar_vec, 3, axis=-1)
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = jnp.arange(len(s1))
                index_list.append(tmp_index)
            else:
                key = random.PRNGKey(i)
                ran_idx = random.permutation(key, len(s1)) 
                index_list.append(ran_idx)
        res_index = jnp.concatenate(index_list)
        t1_all = t1[res_index] 
        t2_all = t2[res_index]
        t3_all = t3[res_index]
        t_all = jnp.concatenate((t1_all, t2_all, t3_all),axis = -1)
        s1_all = s1[res_index] 
        s2_all = s2[res_index]
        s3_all = s3[res_index]
        s_all = jnp.concatenate((s1_all, s2_all, s3_all),axis = -1)
        tar_patch = t_all.T.reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = s_all.T.reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss










