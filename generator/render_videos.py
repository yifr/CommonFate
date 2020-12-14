import os
import sys
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pyquaternion import Quaternion

def render_frame(mesh, rotation_axis=None, degrees=0, transformation_matrix=None):
    """
    Render individual frame, return figure
    Params:
        mesh: :obj: : Trimesh object
        rotation_axis: :np.array: : 1x3 array describing scale of rotation around x,y,z axis
        degrees: :float: : How many degrees of rotation
        transformation_matrix: :np.array: : 4x4 matrix describing pose (taken from quaternion if None)

    Returns:
        Figure of mesh at specific pose and assorted kwargs

    """
    tmesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=[500, 500, 500, 1000])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

    c = 2**-0.5
    pose = [[ 1,  0,  0,  0],
            [ 0,  c, -c, -3],
            [ 0,  c,  c,  3],
            [ 0,  0,  0,  1]]

    mesh_node = scene.add(tmesh, pose=np.eye(4), name='mesh')
    camera_node = scene.add(camera, pose=pose)
    r = pyrender.OffscreenRenderer(640, 480)

    # Define Quaternion for transformation matrix
    if not rotation_axis:
        rotation_axis = np.random.uniform(-1, 1, size=3)
    q = Quaternion(axis=rotation_axis, degrees=degrees)
    if not transformation_matrix:
        transformation_matrix = q.transformation_matrix

    scene.update_pose('mesh', transformation_matrix)

    # Plot and delete renderer
    fig = plt.figure()
    color, depth = r.render(scene)
    plt.imshow(color)
    plt.axis('off')
    r.delete()

    return fig, {'transformation_matrix': transformation_matrix, 'axis': rotation_axis, 'degrees': degrees}

def render_frame_sequence(shape, scene_dir, n_frames=200):
    """
    Renders N frames defining a 360 degree rotation about a random axis
    Saves individual frame for each step in the rotation
    """
    img_dir = scene_dir + '/images'
    tmesh = pyrender.Mesh.from_trimesh(shape)
    bg_color =
    scene = pyrender.Scene(ambient_light=[500, 500, 500, 1000], bg_color=scene_dir + '/texture.jpg')
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

    c = 2**-0.5
    pose = [[ 1,  0,  0,  0],
            [ 0,  c, -c, -3],
            [ 0,  c,  c,  3],
            [ 0,  0,  0,  1]]

    mesh_node = scene.add(tmesh, pose=np.eye(4), name='mesh')
    camera_node = scene.add(camera, pose=pose)

    x = 640
    y = 480
    r = pyrender.OffscreenRenderer(x, y)
    rotation_axis = np.random.uniform(-1, 1, size=3)
    data = {'frames': np.zeros(shape=[n_frames, y, x, 3]),
            'rotation': np.zeros(shape=[n_frames, 3, 3]),
            'quaternion': np.zeros(shape=[n_frames, 4]),
            'angle': np.zeros(shape=[n_frames, 1]),
            'axis': np.zeros(shape=[n_frames, 3])}


    fig = plt.figure()
    rotation_axis = np.random.uniform(-1, 1, 3)
    degrees = np.linspace(0, 360, n_frames)

    for i, d in tqdm(enumerate(degrees)):
        q = Quaternion(axis=rotation_axis, degrees=d)
        scene.set_pose(mesh_node, q.transformation_matrix)
        color, depth = r.render(scene)

        data['frames'][i] = color
        data['rotation'][i] = q.rotation_matrix
        data['quaternion'][i] = q.elements
        data['angle'][i] = q.angle
        data['axis'][i] =  q.axis

        plt.imshow(color)

        plt.axis('off')
        plt.savefig(img_dir + '/fig_%d.png'%i)
        plt.clf()

    plt.close()
    r.delete()

    fig = plt.figure()

    return data

def render_file(obj_file, img_dir, n_frames):
    shape = trimesh.load(obj_file)
    data = render_frame_sequence(shape, img_dir, n_frames)
    return data

def create_vid(base_dir, output_name, frame_rate=25):
    ffmpeg_cmd = 'ffmpeg -framerate {} -i {}/images/fig_%d.png -vf format=yuv420p {}/{}'.format(frame_rate, base_dir, base_dir, output_name)
    res = os.system(ffmpeg_cmd)
    if res == 0:
        return True
    else:
        return False

def main():
    n_scenes = 101
    n_frames = 200
    for scene in range(n_scenes, n_scenes+1):
        base_dir = 'objects/scene_%03d'%scene
        obj_file = base_dir + '/textured.obj'

        print(base_dir)
        shape = trimesh.load(obj_file)
        img_dir = os.path.join(base_dir, 'images')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        data = render_frame_sequence(shape, img_dir, n_frames=200)
        np.save(base_dir + '/data.npy', data)

        ret = create_vid(base_dir, 'vid.mp4')
        if not ret:
            print("Error creating ", base_dir)
            sys.exit(0)

if __name__=='__main__':
    main()
