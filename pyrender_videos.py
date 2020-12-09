import os
import sys
import numpy as np
import pyrender
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion

from xvfbwrapper import Xvfb

vdisplay = Xvfb()
vdisplay.start()

try:
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
        tmesh = pyrender.Mesh.from_trimesh(shape)
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
        rotation_axis = np.random.uniform(-1, 1, size=3)
        data = {'frames': np.array(),
                'rotation': np.array(),
                'quaternion': np.array(),
                'angle': np.array(),
                'axis': np.array()}


        fig = plt.figure()
        rotation_axis = np.random.uniform(-1, 1, 3)
        degrees = np.linspace(0, 360, n_frames)

        for i, d in tqdm(enumerate(degrees)):
            q = Quaternion(axis=rotation_axis, degrees=d)
            scene.update_pose('mesh', q.transformation_matrix)
            color, depth = r.render(scene)

            data['frames'].append(color, axis=0)
            data['rotation'].append(q.rotation_matrix, axis=0)
            data['quaternion'].append(q.elements, axis=0)
            data['angle'].append(q.angle, axis=0)
            data['axis'].append(q.axis, axis=0)

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
        n_scenes = 1
        n_frames = 200
        for scene in range(n_scenes):
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
finally:
    vdisplay.stop()
