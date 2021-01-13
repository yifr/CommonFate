import os
import bpy
import time
import datetime
import numpy as np
from pyquaternion import Quaternion

# Set up logger
import logging
FORMAT = '%(asctime)-10s %(message)s'
logging.basicConfig(filename='render_logs', level=logging.INFO, format=FORMAT)

# Add the generator directory to the path for relative Blender imports
import sys
import pathlib

generator_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(generator_path)

import utils
import superquadrics
import BlenderArgparse
from textures import dot_texture
print(os.getcwd())


ROOT_SCENE_DIR = '/om2/user/yyf/CommonFate/data/'

class RenderEngine:
    def __init__(self, scene, device='CUDA', engine='CYCLES', render_size=512, samples=256):
        self.scene = scene
        self.device = device
        self.engine = engine
        self.render_size = render_size
        self.samples = samples

        if self.device == 'CUDA':
            self.activated_gpus = self.enable_gpus(scene)
            print(f'Using following GPUs: {self.activated_gpus}')
    
    def render(self, output_dir):
        """
        Renders the video to a given folder. 
        """
        if output_dir[:-4] != 'img_':
            output_dir = os.path.join(output_dir, 'img_')

        self.set_render_settings()
        scene.render.filepath = output_dir
        bpy.ops.render.render(animation=True)
            
    def set_render_settings(self):
        # Set properties to increase speed of render time
        scene = self.scene
        scene.render.resolution_x = self.render_size
        scene.render.resolution_y = self.render_size
        scene.render.image_settings.color_mode = 'BW'
        scene.render.image_settings.compression = 0
        scene.cycles.samples = self.samples
    
    def enable_gpus(self, scene, device_type='CUDA', use_cpus=False):
        """
        Sets device as GPU and adjusts rendering tile size accordingly
        """
        scene.render.engine = self.engine  # use cycles for headless rendering

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cuda_devices, opencl_devices = cycles_preferences.get_devices()

        if device_type == 'CUDA':
            devices = cuda_devices
        elif device_type == 'OPENCL':
            devices = opencl_devices
        else:
            raise RuntimeError("Unsupported device type")

        activated_gpus = []

        for device in devices:
            if device.type == "CPU":
                device.use = use_cpus
            else:
                device.use = True
                activated_gpus.append(device.name)

        scene.cycles.device = "GPU"
        cycles_preferences.compute_device_type = device_type

        scene.render.tile_x = 256
        scene.render.tile_y = 256

        return activated_gpus

class BlenderScene(object):
    def __init__(self, scene_dir, 
                        shape_params=None, 
                        device='CUDA', 
                        engine='CYCLES',
                        render_size=512,
                        samples=256
                        ):
        self.scene_dir = scene_dir
        self.data = bpy.data
        self.context = bpy.context
        self.shape_params = shape_params
        self.renderer = RenderEngine(self.scene, device, engine, render_size, samples)
        self.rotation_data = os.path.join(self.scene_dir, 'data.npy')

    @property
    def objects(self):
        return self.scene.objects

    @property
    def scene(self):
        return bpy.data.scenes['Scene']

    def delete_all(self, obj_type):
        """
        Deletes all instances of a given object type:
        obj_type: "MESH" | "LIGHT" | "CAMERA" ...
        """
        for obj in self.objects():
            if obj.type == obj_type:
                obj.select_set(True)
            else:
                obj.select_set(False)
        self.set_mode('OBJECT', object_type)
        bpy.ops.object.delete()

    def set_mode(self, mode, obj_type='MESH'):
        for obj in self.objects():
            if obj.type == obj_type:
                self.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode=mode)

    def generate_mesh(self, shape_params=None):
        """
        Returns points for a superquadric given some shape parameters
        """
        if not self.shape_params or shape_params:
            self.shape_params = np.random.randint(0, 4, 3)
            self.shape_params[2] = self.shape_params[0]
            
            # Save parameters
            self.shape_param_file = os.path.join(self.scene_dir, 'params.txt')
            with open(self.shape_param_file, 'w') as f:
                f.write(f'e1: {self.shape_params[0]:3f}\ne2: {self.shape_params[1]:3f}\ne3: {self.shape_params[2]:3f}')

        n_points = 100
        x, y, z = superquadrics.superellipsoid(self.shape_params, [1, 1, 1, 1], n_points)

        return x, y, z

    def add_mesh(self):
        """
        Adds a mesh from shape parameters
        """
        x, y, z = self.generate_mesh()
        faces, verts = superquadrics.get_faces_and_verts(x, y, z)
        edges = []
        
        mesh = self.data.meshes.new('mesh')
        obj = self.objects().new(mesh.name, mesh)
        col = self.data.collections.get('Collection')
        col.objects.link(obj)

        self.context.view_layer.objects.active = obj
        mesh.from_pydata(verts, edges, faces)

        return mesh
        
    def load_mesh(self, save=False):
        mesh_file = os.path.join(self.scene_dir, 'mesh.obj')
        
        if not os.path.exists(mesh_file):
            x, y, z = self.generate_mesh()
            superquadrics.save_obj_not_overlap(mesh_file, x, y, z)

        mesh = bpy.ops.import_scene.obj(filepath=mesh_file)
        return mesh    

    def texture_mesh(self, obj=None):
        """
        Add texture to a given object. If obj==None, take the active object
        """
        self.set_mode('EDIT')
        texture_file = os.path.join(self.scene_dir, 'texture.png')
        
        # Export new random texture if it doesn't exist
        if not os.path.exists(texture_file):
            min_dot_diam = np.random.randint(10, 15)
            max_dot_diam = np.random.randint(30, 45)
            n_dots = np.random.randint(100, 300)
            texture = dot_texture(min_diameter=min_dot_diam,
                              max_diameter=max_dot_diam,
                              n_dots=n_dots,
                              save_file=texture_file)

        image = self.data.images.load(filepath=texture_path)
        bpy.ops.uv.cube_project(cube_size=1)

        # Create new texture slot
        mat = self.data.materials.new(name="texture")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodeOut = nodes['Material Output']

        # Add emission shader node
        nodeEmission = nodes.new(type='ShaderNodeEmission')
        nodeEmission.inputs['Strength'].default_value = 10

        # Add image to texture
        texImage = nodes.new('ShaderNodeTexImage')
        texImage.image = image

        # Link everything together
        links = mat.node_tree.links
        linkTexture = links.new(texImage.outputs['Color'], nodeEmission.inputs['Color'])
        linkOut = links.new(nodeEmission.outputs['Emission'], nodeOut.inputs['Surface'])

        # Turn off a bunch of material parameters
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Specular'].default_value = 0
        bsdf.inputs['Specular Tint'].default_value = 0
        bsdf.inputs['Roughness'].default_value = 0
        bsdf.inputs['Sheen Tint'].default_value = 0
        bsdf.inputs['Clearcoat'].default_value = 0
        bsdf.inputs['Subsurface Radius'].default_value = [0, 0, 0]
        bsdf.inputs['IOR'].default_value = 0
        mat.cycles.use_transparent_shadow = False

        if obj == None:
            # Add material to active object
            obj = self.context.view_layer.objects.active
        
        # Assign texture to object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


    def set_background_color(self, color=(255,255,255,1)):
        """
        color should be a tuple (R,G,B,alpha) where alpha controls 
        the transparency. Default color is white
        """
        # Set transparent background for render
        scene = self.scene
        scene.render.film_transparent = True
        scene.view_settings.view_transform = 'Raw'

        # Add alpha over node for colored background
        scene.use_nodes = True
        node_tree = scene.node_tree
        alpha_over = node_tree.nodes.new('CompositorNodeAlphaOver')
        
        alpha_over.inputs[1].default_value = color        
        alpha_over.use_premultiply = True
        alpha_over.premul = 1

        # Connect alpha over node
        render_layers = node_tree.nodes['Render Layers']
        composite = node_tree.nodes['Composite']
        node_tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
        node_tree.links.new(alpha_over.outputs['Image'], composite.inputs['Image'])


    def set_light_source(self, light_type, location, rotation):
        """
        Creates a single light source at a specific location / rotation

        Parameters
        -----------
        type: Light type ('POINT' | 'SUN' | 'SPOT' | 'AREA')
        location: Vector location of light
        rotation: Euler angle rotation of light
        """
        light_type = light_type.upper()
        utils.delete_all(obj_type='LIGHT') # Delete existing lights
        light_data = bpy.data.lights.new(name='Light', type=light_type)
        light_data.energy = 10
        light_data.specular_factor = 0
        light_object = bpy.data.objects.new(name='Light', object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object

        light_object.location = location
        light_object.rotation_euler = rotation

    def generate_random_rotation(self, n_frames=100, save=True):
        data = {'quaternion': np.zeros(shape=[n_frames, 4]),
                'rotation': np.zeros(shape=[n_frames, 3, 3]),
                'angle': np.zeros(shape=[n_frames, 1]),
                'axis': np.zeros(shape=[n_frames, 3]),
                'translation': np.zeros(shape=[n_frames, 4, 4]),
                }
        
        rotation_axis = np.random.uniform(-1, 1, 3)
        degrees = np.linspace(0, 360, n_frames)
       
        for frame, degree in enumerate(degrees):
            q = Quaternion(axis=rotation_axis, degrees=degree)

            # Record params
            data['quaternion'][frame] = q.elements
            data['rotation'][frame] = q.rotation_matrix
            data['angle'][frame] = degree
            data['axis'][frame] = rotation_axis

        if save:
            np.save(self.rotation_data, data, allow_pickle=True)

        return data

    def rotate(self, obj, rotations, n_frames=100):
        """
        Takes a mesh and animates a rotation. Generates rotation data 
        if no rotation parameters are passed
        """
        # Set scene parameters
        scene = self.scene
        scene.frame_start = 1
        scene.frame_end = n_frames

        if rotations == None:
            if os.path.exists(self.rotation_data):
                data = np.load(self.rotation_data, allow_pickle=True).item()
            else:
                data = self.generate_random_rotation(n_frames=100, save=True)
            rotations = data['quaternion']

        # Set rotation parameters
        obj.rotation_mode = 'QUATERNION'

        # Add frame for rotation
        for q in rotations:  
            obj.rotation_quaternion = q
            obj.keyframe_insert('rotation_quaternion', frame=frame + 1)


def main(args):
    if not os.path.exists(args.root_dir):
        os.mkdir(args.root_dir)
        print('Created root directory: ', args.root_dir)

    #################################
    # Delete default cube and light
    #################################
    utils.set_mode('OBJECT')
    utils.delete_all(obj_type='MESH')
    utils.delete_all(obj_type='LIGHT')

    # Set white ambient light
    camera_loc = bpy.data.objects['Camera'].location
    camera_rot = bpy.data.objects['Camera'].rotation_euler
    scene.set_light_source('SUN', camera_loc, camera_rot)

    ############################
    # Set high ambient light
    ############################
    world_nodes = bpy.data.worlds['World'].node_tree.nodes
    world_nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    world_nodes['Background'].inputs['Strength'].default_value = 1.5

    for scene_num in range(args.start_scene, args.start_scene + args.n_scenes):
        scene_dir = os.path.join(args.root_dir, 'scene_%03d' % scene_num)
        logging.info('Processing scene: {}...'.format(scene_dir))

        ###################
        # Set paths
        ###################
        image_dir = os.path.join(scene_dir, 'images')
        obj_file = os.path.join(scene_dir, 'mesh.obj')
        data_file = os.path.join(scene_dir, 'data.npy')
        texture_file = os.path.join(scene_dir, 'texture.png')

        ####################################################
        # Create random dot texture image and save to a file
        #####################################################
        texture = dot_texture(min_diameter=args.min_dot_diam,
                              max_diameter=args.max_dot_diam,
                              n_dots=args.n_dots,
                              save_file=texture_file)

        #############################
        # Load texture image and mesh
        #############################
        img = utils.load_img(texture_file)
        mesh = utils.load_obj(obj_file)
        mesh_name = bpy.context.selected_objects[0].name

        ###########################################
        # Wrap texture and save textured mesh
        ############################################
        wrap_texture(img)
        utils.export_obj(mesh, scene_dir)

        ######################################
        # Set material properties
        #######################################
        mat = bpy.data.materials[-1]
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Specular'].default_value = 0
        bsdf.inputs['Specular Tint'].default_value = 0
        bsdf.inputs['Roughness'].default_value = 0
        bsdf.inputs['Sheen Tint'].default_value = 0
        bsdf.inputs['Clearcoat'].default_value = 0
        bsdf.inputs['Subsurface Radius'].default_value = [0, 0, 0]
        bsdf.inputs['IOR'].default_value = 0
        mat.cycles.use_transparent_shadow = False

        ################################
        # Set global light properties
        ###############################
        scene = bpy.data.scenes[-1]
        scene.render.engine = 'CYCLES'
        light = bpy.data.lights[-1]
        light.cycles['cast_shadow'] = False

        ######################
        # Animate rotation
        ######################
        obj = bpy.data.objects[mesh_name]
        data = rotate(obj)

        #######################################################
        # Save rotation parameters and mesh, and render images
        #######################################################
        np.save(data_file, data)
        render(args, output_dir=image_dir)

        utils.delete_all(obj_type='MESH')

        # Add a render timestamp
        ts = time.time()
        fts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(scene_dir, 'timestamp.txt'), 'w') as f:
            f.write('Files Rendered at: {}\n'.format(fts))


        logging.info('...Done')

if __name__=='__main__':
    parser = BlenderArgparse.ArgParser()
    parser.add_argument('--n_scenes', type=int, help='Number of scenes to generate', default=1000)
    parser.add_argument('--root_dir', type=str, help='Output directory for data', default='scenes')
    parser.add_argument('--render_size', type=int, help='size of .png file to render', default=1024)
    parser.add_argument('--n_frames', type=int, help='Number of frames to render per scene', default=100)
    parser.add_argument('--device', type=str, help='Either "cuda" or "cpu"', default='cuda')
    parser.add_argument('--start_scene', type=int, help='Scene number to begin rendering from', default=0)

    # Texture default settings:
    min_dot_diam = np.random.randint(5, 15)
    max_dot_diam = np.random.randint(30, 45)
    n_dots= np.random.randint(100, 300)
    parser.add_argument('--min_dot_diam', type=int, help='minimum diamater for dots on texture image', default=min_dot_diam)
    parser.add_argument('--max_dot_diam', type=int, help='maximum diamater for dots on texture image', default=max_dot_diam)
    parser.add_argument('--n_dots', type=int, help='Number of dots on texture image', default=n_dots)
    parser.add_argument('--output_img_name', type=str, help='Name for output images', default='img')
    parser.add_argument('--texture_only', action='store_true', help='Will output only textured mesh, and no rendering')

    args = parser.parse_args()
    if args.device == 'cuda':
        enable_gpus('CUDA')

    main(args)
