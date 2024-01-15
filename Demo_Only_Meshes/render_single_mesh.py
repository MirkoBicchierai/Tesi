import argparse
import os
import cv2
import numpy as np
import pyrender
import trimesh
from psbody.mesh import Mesh

# os.environ['PYOPENGL_PLATFORM'] = 'egl'


def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m',
                       min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):
    background_black = False
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


def render_mesh(vertices, template, save_path, image_name, uv_template_fname='', texture_img_fname=''):
    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:, :, ::-1]
    else:
        vt, ft = None, None
        tex_img = None

    center = np.mean(vertices, axis=0)

    render_mesh = Mesh(vertices, template.faces)
    if vt is not None and ft is not None:
        render_mesh.vt, render_mesh.ft = vt, ft
    img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
    cv2.imwrite(os.path.join(save_path, image_name), img[:, 100:-100])


def main():
    parser = argparse.ArgumentParser(description='Python file to render a mesh into an image')
    parser.add_argument("--save_path", type=str, default='img', help='path for video')
    parser.add_argument("--mesh_path", type=str,
                        default="videoCOMA/WHITE/video30/GeneratedVideo_FaceTalk_170811_03274_TA_mouth-extreme/Meshes/tst039.ply",
                        help='path for the meshes sequence')

    # videoCOMAFlorence/WHITE/video30/GeneratedVideo_CH02_Kissy/Meshes/tst059.ply
    parser.add_argument("--flame_template", type=str,
                        default="S2D/template/flame_model/FLAME_sample.ply",
                        help='template_path')
    parser.add_argument("--image_name", type=str, default='mouth-extreme_FaceTalk_170811_03274_TA_TA_6.png', help='name of the image')
    # Kissy_CH02_6.png
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    template = trimesh.load(args.flame_template, process=False)
    mesh = trimesh.load(args.mesh_path, process=False)

    print('Image Generation')

    render_mesh(mesh.vertices,
                template,
                args.save_path,
                args.image_name)

    print('done')


if __name__ == '__main__':
    main()
