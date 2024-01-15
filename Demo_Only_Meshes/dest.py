import sys
import S2D.models as models
import S2D.spiral_utils as spiral_utils
import S2D.shape_data as shape_data
import S2D.autoencoder_dataset as autoencoder_dataset
import S2D.save_meshes as save_meshes
import argparse
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from my_test_funcs import test_autoencoder_dataloader
import torch

import Get_landmarks as Get_landmarks
import time
import os
import cv2
import tempfile
import numpy as np
from subprocess import call
from psbody.mesh import Mesh
import pyrender
import trimesh
import matplotlib as mpl
import matplotlib.cm as cm
import glob
# import librosa
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')

def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):

    background_black = True
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v - t_center).T).T + t_center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material, smooth=True)

    if background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])  # [0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2],
                               bg_color=[255, 255, 255])  # [0, 0, 0] black,[255, 255, 255] white

    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                       fy=camera_params['f'][1],
                                       cx=camera_params['c'][0],
                                       cy=camera_params['c'][1],
                                       znear=frustum['near'],
                                       zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(sequence_vertices, template, out_path , out_fname, fps, uv_template_fname='', texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 800), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    i = 0
    for i_frame in range(num_frames - 2):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        writer.write(img)
        #plt.savefig('/home/federico/Scrivania/Universita/TESI/generated/' + 'image' + str(i))
        i = i + 1
    writer.release()

    video_fname = os.path.join(out_path, out_fname)
    cmd = ('ffmpeg' + ' -i {0} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -ar 22050 {1}'.format(
     tmp_video_file.name, video_fname)).split()
    call(cmd)

def generate_mesh_video(out_path, out_fname, meshes_path_fname, fps, template):

    sequence_fnames = sorted(glob.glob(os.path.join(meshes_path_fname, '*.ply*')))

    uv_template_fname = template
    sequence_vertices = []
    f = None

    for frame_idx, mesh_fname in enumerate(sequence_fnames):
        frame = Mesh(filename=mesh_fname)
        sequence_vertices.append(frame.v)
        if f is None:
            f = frame.f

    template = Mesh(sequence_vertices[0], f)
    sequence_vertices = np.stack(sequence_vertices)
    render_sequence_meshes(sequence_vertices, template, out_path, out_fname, fps, uv_template_fname=uv_template_fname, texture_img_fname='')


def elaborate_landmarks(landmarks, actor_landmarks, actor_vertices, save_path):

    if not os.path.exists(os.path.join(save_path, 'points_input')):
        os.makedirs(os.path.join(save_path, 'points_input'))

    if not os.path.exists(os.path.join(save_path, 'points_target')):
        os.makedirs(os.path.join(save_path, 'points_target'))

    if not os.path.exists(os.path.join(save_path, 'landmarks_target')):
        os.makedirs(os.path.join(save_path, 'landmarks_target'))

    if not os.path.exists(os.path.join(save_path, 'landmarks_input')):
        os.makedirs(os.path.join(save_path, 'landmarks_input'))

    for j in range(len(landmarks)):
                np.save(os.path.join(save_path, 'points_input', '{0:08}_frame'.format(j)), actor_vertices)
                np.save(os.path.join(save_path, 'points_target', '{0:08}_frame'.format(j)), actor_vertices)
                np.save(os.path.join(save_path, 'landmarks_target', '{0:08}_frame'.format(j)), landmarks[j])
                np.save(os.path.join(save_path, 'landmarks_input', '{0:08}_frame'.format(j)), actor_landmarks)

    files = []

    for r, d, f in os.walk(os.path.join(save_path, 'points_input')):
                for file in f:
                    if '.npy' in file:
                        files.append(os.path.splitext(file)[0])
    np.save(os.path.join(save_path, 'paths_test.npy'), sorted(files))

    files = []
    for r, d, f in os.walk(os.path.join(save_path, 'landmarks_target')):
        for file in f:
            if '.npy' in file:
                files.append(os.path.splitext(file)[0])
    np.save(os.path.join(save_path, 'landmarks_test.npy'), sorted(files))
    return sorted(files)

def generate_meshes_from_landmarks(template_path, reference_mesh_path, landmarks_path, save_meshes_path, s2d_model_path):
    filter_sizes_enc = [[3, 16, 32, 64, 128], [[], [], [], [], []]]
    filter_sizes_dec = [[128, 64, 32, 32, 16], [[], [], [], [], 3]]
    nz = 16
    ds_factors = [4, 4, 4, 4]
    reference_points = [[3567, 4051, 4597]]
    nbr_landmarks = 68
    step_sizes = [2, 2, 1, 1, 1]
    dilation = [2, 2, 1, 1, 1]
    device_idx = 0
    torch.cuda.get_device_name(device_idx)

    meshpackage = 'trimesh'

    shapedata = shape_data.ShapeData(nVal=100,
                                     test_file=landmarks_path + '/test.npy',
                                     reference_mesh_file=reference_mesh_path,
                                     normalization=False,
                                     meshpackage=meshpackage, load_flag=False)

    shapedata.n_vertex = 5023
    shapedata.n_features = 3

    with open(
            './S2D/template/template/COMA_downsample/downsampling_matrices.pkl',
            'rb') as fp:
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i in
         range(len(M_verts_faces))]

    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

    for i in range(len(ds_factors)):
        dist = euclidean_distances(M[i + 1].vertices, M[0].vertices[reference_points[0]])
        reference_points.append(np.argmin(dist, axis=0).tolist())

    Adj, Trigs = spiral_utils.get_adj_trigs(A, F, shapedata.reference_mesh, meshpackage='trimesh')

    spirals_np, spiral_sizes, spirals = spiral_utils.generate_spirals(step_sizes,
                                                                      M, Adj, Trigs,
                                                                      reference_points=reference_points,
                                                                      dilation=dilation, random=False,
                                                                      meshpackage='trimesh',
                                                                      counter_clockwise=True)

    sizes = [x.vertices.shape[0] for x in M]

    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

    tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]

    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        bD.append(d)
        bU.append(u)

    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]

    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    dataset_test = autoencoder_dataset.autoencoder_dataset(neutral_root_dir=landmarks_path, points_dataset='test',
                                                           shapedata=shapedata,
                                                           normalization=False, template=reference_mesh_path)

    dataloader_test = DataLoader(dataset_test, batch_size=1,
                                 shuffle=False, num_workers=4)

    model = models.SpiralAutoencoder(filters_enc=filter_sizes_enc,
                                     filters_dec=filter_sizes_dec,
                                     latent_size=nz,
                                     sizes=sizes,
                                     nbr_landmarks=nbr_landmarks,
                                     spiral_sizes=spiral_sizes,
                                     spirals=tspirals,
                                     D=tD, U=tU, device=device).to(device)


    checkpoint = torch.load(s2d_model_path, map_location=device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])

    predictions, inputs, lands, targets = test_autoencoder_dataloader(device, model, dataloader_test, shapedata)
    np.save(os.path.join(save_meshes_path, 'targets'), targets)
    np.save(os.path.join(save_meshes_path, 'predictions'), predictions)
    save_meshes.save_meshes(predictions, save_meshes_path, n_meshes=len(predictions), template_path=template_path)
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Landmarks2Meshes')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--actor", type=str, default="/mnt/diskone-first/mbicchierai/COMA_Florence_Templates/COMA_CH01.ply")
    parser.add_argument("--landmarks_path", type=str, default=None, help='path of the generated_landmarks')
    parser.add_argument("--s2d_model", type=str, default="/mnt/diskone-first/mbicchierai/s2d_coma_florence.pth.tar", help='path of the s2d model')
    parser.add_argument("--save_path", type=str, default="Example_COMA_FLAME", help='path to save')

    meshes_path = '/diskone-first/mbicchierai/COMA_FLAME/COMA_CH01/Afraid'
    landmarks = []
    for i in range(61):
        mesh = trimesh.load(os.path.join(meshes_path, 'Afraid_' + str(i).zfill(2) + '.ply'), process=False)
        landmarks.append(Get_landmarks.get_landmarks(mesh.vertices))
    landmarks = np.array(landmarks)
    args = parser.parse_args()
    #landmarks = np.load(args.landmarks_path)
    #landmarks = np.reshape(landmarks, (len(landmarks), 68, 3))
    actor_mesh = trimesh.load(args.actor, process=False)
    #actor_mesh = trimesh.load(actor_mesh_path)
    #actor_vertices = actor_mesh.vertices
    actor_landmarks = Get_landmarks.get_landmarks(actor_mesh.vertices)
    save_path = args.save_path
    os.mkdir(os.path.join(save_path))
    os.mkdir(os.path.join(save_path, 'Meshes'))
    os.mkdir(os.path.join(save_path, 'Landmarks'))

    template_path = './S2D/template/flame_model/FLAME_sample.ply'
    save_path_meshes = os.path.join(save_path, 'Meshes')
    save_landmarks_path = os.path.join(save_path, 'Landmarks')
    print('Landmarks Elaboration')
    elaborate_landmarks(landmarks, actor_landmarks, actor_mesh.vertices, save_landmarks_path)

    print('Meshes Generation')
    generate_meshes_from_landmarks(template_path, template_path, save_landmarks_path, save_path_meshes, args.s2d_model)
    end = time.time()

    save_video_path = os.path.join(save_path)

    print('Video Generation')
    generate_mesh_video(save_video_path,
                        'example.mp4',
                        save_path_meshes,
                        60,
                        template_path)
    print('done')
