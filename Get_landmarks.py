"""
modified from https://github.com/soubhiksanyal/RingNet/blob/master/dynamic_contour_embedding.py
"""

import numpy as np
import trimesh
import _pickle as pickle
# mport matplotlib.pyplot as plt
import time


def load_static_embedding(static_embedding_path):
    with open(static_embedding_path, 'rb') as f:
        lmk_indexes_dict = pickle.load(f, encoding='latin1')
    lmk_face_idx = lmk_indexes_dict['lmk_face_idx'].astype(np.uint32)
    lmk_b_coords = lmk_indexes_dict['lmk_b_coords']
    return lmk_face_idx, lmk_b_coords


def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
    # function: evaluation 3d points given mesh and landmark embedding
    # modified from https://github.com/Rubikplayer/flame-fitting/blob/master/fitting/landmarks.py
    dif1 = np.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                      (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                      (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1


def load_dynamic_contour(vertices, faces, contour_embeddings_path='None', static_embedding_path='None', angle=0):
    # template_mesh = trimesh.load(template_flame_path, process=False)   #Mesh(filename=template_flame_path)
    contour_embeddings_path = contour_embeddings_path
    dynamic_lmks_embeddings = np.load(contour_embeddings_path, allow_pickle=True, encoding='latin1').item()
    lmk_face_idx_static, lmk_b_coords_static = load_static_embedding(static_embedding_path)
    lmk_face_idx_dynamic = dynamic_lmks_embeddings['lmk_face_idx'][angle]
    lmk_b_coords_dynamic = dynamic_lmks_embeddings['lmk_b_coords'][angle]
    dynamic_lmks = mesh_points_by_barycentric_coordinates(vertices, faces, lmk_face_idx_dynamic, lmk_b_coords_dynamic)
    static_lmks = mesh_points_by_barycentric_coordinates(vertices, faces, lmk_face_idx_static, lmk_b_coords_static)
    total_lmks = np.vstack([dynamic_lmks, static_lmks])

    ### for visualization
    #     fig = plt.figure()
    #     ax = plt.axes(projection="3d")
    #     ax.plot3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'g*')
    #     ax.plot3D(total_lmks[:, 0], total_lmks[:, 1], total_lmks[:, 2],  'r*')
    #     plt.show()
    #     time.sleep(5000)
    return total_lmks


def get_landmarks(vertices):
    angle = 0.0
    if angle < 0:
        angle = 39 - angle
    contour_embeddings_path = 'S2D/template/flame_model/flame_dynamic_embedding.npy'
    static_embedding_path = 'S2D/template/flame_model/flame_static_embedding.pkl'

    try:
        template_mesh = trimesh.load('S2D/template/template/template.obj',
                                     process=False)
    except:
        contour_embeddings_path = '../S2D/template/flame_model/flame_dynamic_embedding.npy'
        static_embedding_path = '../S2D/template/flame_model/flame_static_embedding.pkl'
        template_mesh = trimesh.load('../S2D/template/template/template.obj',
                                     process=False)

    faces = template_mesh.faces
    total_lmks = load_dynamic_contour(vertices, faces, contour_embeddings_path=contour_embeddings_path,
                                      static_embedding_path=static_embedding_path, angle=int(angle))
    return total_lmks
