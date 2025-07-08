import trimesh
import numpy as np
import time
import json

with open('source/test_buoyancy/asset/masks/config.json', 'r') as f:
    config = json.load(f)

voxel_res = config.get("resolution")
barge_mesh_file = config.get("barge_mesh_file")

mesh = trimesh.load(barge_mesh_file)
bounds = mesh.bounds
min_bound, max_bound = bounds[0], bounds[1]
interval = (max_bound - min_bound) / voxel_res
xs = np.arange(min_bound[0], max_bound[0], interval[0])
ys = np.arange(min_bound[1], max_bound[1], interval[1])
zs = np.arange(min_bound[2], max_bound[2], interval[2])
xv, yv, zv = np.meshgrid(xs, ys, zs, indexing='ij')
voxel_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
t_start = time.time()
inside_mask = mesh.contains(voxel_centers)
t_end = time.time()
print("Mask generated, time used: ", t_end-t_start)

valid_voxels = voxel_centers[inside_mask]
voxel_volume = interval[0] * interval[1] * interval[2]


np.save("source/test_buoyancy/asset/masks/barge_inside_mask_barge.npy", inside_mask)
np.save("source/test_buoyancy/asset/masks/barge_valid_voxels.npy", valid_voxels)
np.save("source/test_buoyancy/asset/masks/barge_voxel_volume.npy", voxel_volume)