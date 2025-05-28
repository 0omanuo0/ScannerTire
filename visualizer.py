import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt


import trimesh
import cv2

def export_to_stl(file_name, stl_file_name):
    data = np.load(file_name)
    if data.shape != (300, 1296, 1):
        raise ValueError(f"Unexpected array shape: {data.shape}, expected (300, 1296, 1)")
    data = data.reshape(300, 1296)
    data[data > 220] = 0
    data[data < 0] = 0
    rows, cols = data.shape

    # Create grid of (x, y, z) points
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    z = data

    # Flatten the arrays
    vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

    # Create faces (two triangles per grid square)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx = i * cols + j
            # Triangle 1
            faces.append([idx, idx + 1, idx + cols])
            # Triangle 2
            faces.append([idx + 1, idx + cols + 1, idx + cols])
    faces = np.array(faces)

    # Create mesh and export
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(stl_file_name)
    print(f"Exported to {stl_file_name}")
    
    
def export_heightmap_texture(file_name, image_file_name):
    data = np.load(file_name)
    if data.shape != (300, 1296, 1):
        raise ValueError(f"Unexpected array shape: {data.shape}, expected (300, 1296, 1)")
    data = data.reshape(300, 1296)
    data[data > 220] = 0
    data[data < 0] = 0

    # Focus color mapping around value 150
    vmin = 140
    vmax = 160

    plt.imsave(
        image_file_name,
        data,
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        format='png'
    )
    plt.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Heightmap Texture')
    plt.show()
    print(f"Heightmap texture saved as {image_file_name}")



def visualize_3d_array(file_name):
    # Load the .npy file
    data = np.load(file_name)
    # Ensure the array has the expected shape
    shape = data.shape
    # Reshape the array to remove the singleton dimension
    data = data.reshape(shape[0], shape[1])
    data = data[:100]  # get only the first 100 rows

    xshape = data.shape[1]
    yshape = data.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(0, xshape)
    y = np.arange(0, yshape)
    x, y = np.meshgrid(x, y)
    z = data

    # Optional: smooth the surface using Gaussian blur
    z2 = z.copy()
    z2 = cv2.GaussianBlur(z2, (11, 11), sigmaX=0, sigmaY=0)
    z2 = z2.reshape(yshape, xshape)

    # Plot the surface
    ax.plot_surface(x, y, z2, cmap='viridis')


    plt.show()
    
def visualizeImg(file_name):
    # Load the .npy file
    data = np.load(file_name)
    shape = data.shape
    vmin = -220
    vmax = -140
    # Reshape the array to remove the singleton dimension
    data = data.reshape(shape[0], shape[1])
    # data[data > 220] = 0
    # data[data < 0] = 0
    
    # subplot into 2 rows
    # fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    plt.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    
    

    # data1 = []
    # data2 = []
    # for row, i in enumerate(data):
    #     if(row % 2 == 0):
    #         data1.append(i)
    #     else:
    #         data2.append(i)
    # data1 = np.array(data1)
    # data2 = np.array(data2)
    # data = np.concatenate((data1, data2), axis=0)
    
    
    
    # # Create a 2D image plot
    # axs[1].imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    # axs[1].set_title('2D Image from 3D Array')
    # plt.colorbar(axs[1].imshow(data, cmap='jet', vmin=vmin, vmax=vmax), ax=axs[1])
    plt.show()

# Example usage
uuid = "c0cf64e8-dfb1-4085-875e-326191f80bfb"
file_name = f"scans/dz_processed-{uuid}.npy"  # Replace with your .npy file path
# visualizeImg(file_name)
visualize_3d_array(file_name)
# Example usage
# stl_file_name = "output_scan.stl"
# export_heightmap_texture(file_name, "output_scan.png")
# export_to_stl(file_name, stl_file_name)