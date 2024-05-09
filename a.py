import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from multiprocessing import Pool

from pyoints import (
    storage,
    Extent,
    transformation,
    filters,
    registration,
    normals,
)

colors = {'A': 'green', 'B': 'blue', 'C': 'red'}

coords_dict = {}

# Cargar las imágenes de profundidad
for i in range(1, 20):
    depth_image = np.load(f"data\depth_image.npy ({i}).npy")
    pcd = o3d.geometry.PointCloud()
    
    x1 = np.linspace(0, depth_image.shape[1] - 1, depth_image.shape[1])
    y1 = np.linspace(0, depth_image.shape[0] - 1, depth_image.shape[0])
    x1, y1 = np.meshgrid(x1, y1)
    z1 = depth_image.flatten()
    
    points1 = np.vstack((x1.flatten(), y1.flatten(), z1)).T
    coords_dict[f'frame_{i}'] = points1
    
print("Data loaded")

d_thresh = 0.01
radii = [d_thresh, d_thresh, d_thresh]
icp = registration.ICP(
    radii,
    max_iter=60,
    max_change_ratio=0.000001,
    k=1
)

T_dict, pairs_dict, report = icp(coords_dict)

# pointclouds = []

# for key in coords_dict:
#     coords = transformation.transform(coords_dict[key], T_dict[key])
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(coords)
    
#     pointclouds.append(pc)
    

# o3d.visualization.draw_geometries([pointclouds])
    

def transform_and_visualize(key):
    coords = transformation.transform(coords_dict[key], T_dict[key])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(coords)
    o3d.visualization.draw_geometries([pc])

# Suponiendo que coords_dict y T_dict ya están definidos

# Crear un pool de procesos
with Pool(processes=20) as p:
    # Aplicar la función transform_and_visualize a cada clave en coords_dict
    p.map(transform_and_visualize, coords_dict.keys())