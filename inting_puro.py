import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData

def read_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertices = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T
    faces = [list(face) for face in ply_data['face']['vertex_indices']]
    return vertices, faces

def update(frame, ax, mesh_collection, file_paths):
    ax.cla()
    ax.set_title(f'Frame {frame}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    current_file_path = file_paths[frame]
    vertices, faces = read_ply(current_file_path)

    # Create a Poly3DCollection for the mesh
    triangles = [vertices[face] for face in faces]
    mesh_collection.set_verts(triangles)

def main():
    folder_path = 'Demo_Only_Meshes/video40/GeneratedVideo_FaceTalk_170912_03278_TA_cheeks-in/Meshes'
    file_paths = [file for file in os.listdir(folder_path) if file.endswith('.ply')]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection for the mesh
    mesh_collection = Poly3DCollection([], facecolors='b', edgecolors='k', alpha=0.5)
    ax.add_collection3d(mesh_collection)

    animation = FuncAnimation(fig, update, frames=len(file_paths), fargs=(ax, mesh_collection, file_paths), interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()