import os
import sys
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion

def render_img(shape, img_dir, n_frames):
    mesh_node = pyrender.Mesh.from_trimesh(shape)
    scene = pyrender.Scene(ambient_light=[500, 500, 500, 1000])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    #light = pyrender.PointLight(color=[100.0, 100.0, 100.0], intensity=10.0)

    c = 2**-0.5
    pose = [[ 1,  0,  0,  0],
            [ 0,  c, -c, -3],
            [ 0,  c,  c,  3],
            [ 0,  0,  0,  1]]

    mesh_node = scene.add(mesh_node, pose=np.eye(4))
    #scene.add(light, pose=np.eye(4))
    camera_node = scene.add(camera, pose=pose)


    r = pyrender.OffscreenRenderer(640, 480)
    rotation_axis = np.random.uniform(-1, 1, size=3)

    fig = plt.figure()

    rotation_axis = np.random.uniform(-1, 1, 3)
    for i in tqdm(range(n_frames)):
        scene.remove_node(mesh_node)
        shape.vertices = shape.vertices.dot(Quaternion(axis=rotation_axis, degrees=1).rotation_matrix)

        mesh_node = pyrender.Mesh.from_trimesh(shape)
        mesh_node = scene.add(mesh_node, pose=np.eye(4))
        color, depth = r.render(scene)
        plt.imshow(color)

        plt.axis('off')
        plt.savefig(img_dir + '/fig_%d.png'%i)
        plt.clf()

    plt.close()
    r.delete()

    fig = plt.figure()

    return fig, shape


def render_file(obj_file, img_dir, n_frames):
    shape = trimesh.load(obj_file)
    return render_img(shape, img_dir, n_frames)


def create_vid(base_dir, output_name, frame_rate=25):
    ffmpeg_cmd = 'ffmpeg -framerate {} -i {}/images/fig_%d.png -vf format=yuv420p {}/{}'.format(frame_rate, base_dir, base_dir, output_name)
    res = os.system(ffmpeg_cmd)
    if res == 0:
        return True
    else:
        return False


def main():
    n_scenes = 144
    n_frames = 200
    for scene in range(143, n_scenes):
        base_dir = 'objects/scene_%03d'%scene
        obj_file = base_dir + '/textured.obj'

        print(base_dir)
        shape = trimesh.load(obj_file)
        img_dir = os.path.join(base_dir, 'images')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        render_img(shape, img_dir, n_frames=200)

        ret = create_vid(base_dir, 'vid.mp4')
        if not ret:
            print("Error creating ", base_dir)
            sys.exit(0)

if __name__=='__main__':
    main()
