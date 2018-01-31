import matplotlib.pyplot as plt
import numpy as np
import json
import os
import copy

def load_bin_data(fn, dim, dtype):
    with open(fn, 'r') as f:
        d = np.fromfile(f, dtype=dtype)
        dim_ = copy.copy(dim)
        if type(dim) is tuple:
            dim_ = list(dim)
        dim_.reverse()
        data = d.reshape(dim_)
        data = data.transpose()
        return data

with open('14benchmark-internal.json', 'r') as f:
    params = json.load(f)
dim = params['Domain']['Dim']


fn = '14benchmark-internal.mc2'
d = load_bin_data(fn, dim, np.float32)
src = load_bin_data(params['Optode']['Source']['SourceFile'], dim, np.float32)
vol = load_bin_data(params['Domain']['VolumeFile'], dim, np.uint8)

plt.rcParams['figure.subplot.hspace'] = 0.5
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(src[:, :, int(dim[2] / 2)])
axes[0, 1].imshow(vol[int(dim[0] / 2), :, :])
axes[1, 0].imshow(d[:, :, int(dim[2] / 2)])
axes[1, 1].imshow(d[int(dim[0] / 2), :, :])

axes[0, 0].set_title('internal source')
axes[0, 0].set_xlabel('y')
axes[0, 0].set_ylabel('x')
axes[0, 1].set_title('material types')
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('y')
axes[1, 0].set_title('mcx output')
axes[1, 0].set_xlabel('y')
axes[1, 0].set_ylabel('x')
axes[1, 1].set_title('mcx output')
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('y')
plt.show()
