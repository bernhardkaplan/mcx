import numpy as np
import json
import os
import copy

def save_bin_column_major(d, fn, verbose=False):
    dirname = os.path.dirname(fn)
    dim = copy.copy(d.shape)
    if type(dim) is tuple:
        dim = list(dim)
    dim.reverse()
    data = d.transpose()
    data = data.reshape(dim)
    if verbose:
        print('Saving data to: %s with dim %s' % (fn, str(data.shape)))
    f = open(fn, 'wb')
    data.tofile(f)
    f.flush()
    f.close()

# load parameters
with open('14benchmark-internal.json', 'r') as f:
    params = json.load(f)
dim = params['Domain']['Dim']

# create internal source distribution
internal_source = np.zeros(dim, dtype=np.float32)
x0, x1 = int(3 * dim[0]/8), int(5 * dim[0]/8)
y0, y1 = int(3 * dim[1]/8), int(5 * dim[1]/8)
z0, z1 = int(3 * dim[2]/8), int(5 * dim[2]/8)
internal_source[x0:x1, y0:y1, z0:z1] = 1.
output_fn = 'internal_source_60x60x60.bin'
save_bin_column_major(internal_source, output_fn, verbose=True)

# create a block absorber
block_volume = np.ones(dim, dtype=np.uint8)
block_volume[:, :, int(dim[2] / 2): ] = 2
output_fn = 'internal_source_60x60x60.bin'
save_bin_column_major(block_volume, params['Domain']['VolumeFile'], \
        verbose=True)

