import torch
import numpy as np

cavas = torch.zeros(2, 3 * 7, dtype=torch.double)
ny = 7
nx = 3
print(cavas)

this_coords = torch.from_numpy(np.array([
    [0, 4, 1],
    [0, 0, 2],
    [0, 5, 0]
]))

indices = this_coords[:, 2] * ny + this_coords[:, 1]
print(indices)


cavas[:, indices] = torch.from_numpy(np.array(
    [
        [0.9, 2.2, -1.0],
        [8.0, 7.0, -2.2],

    ]))
print(cavas)

batch_canvas = cavas.view(1, 2, nx, ny)

print(batch_canvas)

print(batch_canvas.shape)

batch_canvas = torch.flip(batch_canvas, dims=[2])
print(batch_canvas)
