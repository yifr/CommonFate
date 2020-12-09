import os
import bpy
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

def set_mode(mode, obj_type='MESH'):
    """
    Sets the mode for a specific object type

    Parameters
    -----------
        mode: :str: : Blender mode to set ('EDIT' | 'OBJECT' | etc...)
        type: :str: : Object type to apply setting to
    """
    scene = bpy.context.scene

    for obj in scene.objects:
        if obj.type == obj_type:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode=mode)

def delete_all(obj_type):
    """
    Delete specific type of object

    Parameters
    -----------
        type: name of object type to delete (ie; 'MESH' | 'LIGHT' | 'CAMERA')

    Returns
    ----------
        nothing, but deletes specified objects
    """
    for o in bpy.context.scene.objects:
        if o.type == obj_type:
            o.select_set(True)
        else:
            o.select_set(False)
    set_mode('OBJECT')
    bpy.ops.object.delete()

def cube_project():
    set_mode('EDIT')
    bpy.ops.uv.cube_project(cube_size=1)

def load_img(path):
    img = bpy.data.images.load(filepath=path)
    return img

def load_obj(path):
    mesh = bpy.ops.import_scene.obj(filepath=path)
    return mesh

def export_obj(obj, scene_dir, fname='textured.obj'):
    """
    Exports textured mesh to a file called textured.obj.
    Also exports material group used to texture
    Assumes one active mesh at a time
    """
    set_mode('OBJECT')
    output_file = os.path.join(scene_dir, fname)
    bpy.ops.export_scene.obj(filepath=output_file, use_selection=True)

def dot_texture(width=1024, height=1024, min_diameter=10, max_diameter=20, n_dots=900):
    img  = Image.new('RGB', (width, height), color = 'white')
    draw = ImageDraw.Draw(img)

    for _ in range(n_dots):
        x, y = np.random.randint(0,width), np.random.randint(0,height)
        diam = np.random.randint(min_diameter, max_diameter)
        draw.ellipse([x,y,x+diam,y+diam], fill='black')

    return img


def set_light_source(light_type, location, rotation):
    """
    Creates a single light source at a specific location / rotation

    Parameters
    -----------
        type: Light type ('POINT' | 'SUN' | 'SPOT' | 'AREA')
        location: Vector location of light
        rotation: Euler angle rotation of light
    """
    light_type = light_type.upper()
    delete_all(obj_type='LIGHT') # Delete existing lights
    light_data = bpy.data.lights.new(name='Light', type=light_type)
    light_data.energy = 10
    light_object = bpy.data.objects.new(name='Light', object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object

    light_object.location = location
    light_object.rotation_euler = rotation


def wrap_texture(img):
    """
    Texture by wrapping an image around a mesh
    Assumes only one mesh active at a time
    Params:
        img: :obj: opened blender image
    """
    cube_project()

    # Create new texture slot
    mat = bpy.data.materials.new(name="texture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    # Add image to texture
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = img
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

    ob = bpy.context.view_layer.objects.active

    # Assign texture to object
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)


def rotate(obj, n_frames=100):
    """
    Takes a mesh and animates a rotation of 360 degrees around a random axis

    Params
    ------
        obj: Mesh object to be rotated
        n_frames: how many frames for the video

    Returns
    -------
        Dictionary containing pose information at each step
    """
    data = {'rotation': np.zeros(shape=[n_frames, 4]),
            'angle': np.zeros(shape=[n_frames, 1]),
            'axis': np.zeros(shape=[n_frames, 3]),
            'translation': np.zeros(shape=[n_frames, 4, 4]),
            }

    # Set scene parameters
    scene = bpy.data.scenes['Scene']
    scene.frame_start = 1
    scene.frame_end = n_frames

    # Set rotation parameters
    obj.rotation_mode = 'QUATERNION'
    rotation_axis = np.random.uniform(-1, 1, 3)
    degrees = np.linspace(0, 360, n_frames)

    # Add frame for rotation
    for frame, degree in tqdm(enumerate(degrees)):
        q = Quaternion(axis=rotation_axis, degrees=degree)

        obj.rotation_quaternion = q.elements
        obj.keyframe_insert('rotation_quaternion', frame=frame + 1)

        # Record params
        data['rotation'][frame] = q.elements
        data['angle'][frame] = degree
        data['axis'][frame] = rotation_axis
        data['translation'][frame] = np.array(obj.matrix_world)

    return data

def render(output_dir, add_background=False):
    """
    Renders the video to a given folder. Defaults to adding a white
    background. This can be changed to a textured background in the
    future.

    Params
    ------
        output_dir: Directory to save image files to.
    Returns
    -------
        Nothing, but outputs:
            PNG files for each frame in the scene titled after
            their frame index (ie; img_0001.png, img_0002.png...)
    """
    # Add img_ prefix to output file names
    if output_dir[:-4] != 'img_':
        output_dir = os.path.join(output_dir, 'img_')

    scene = bpy.data.scenes['Scene']

    # Set transparent background for render
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # Add alpha over node for white background
    bpy.context.scene.use_nodes = True
    node_tree = scene.node_tree
    alpha_over = node_tree.nodes.new('CompositorNodeAlphaOver')
    alpha_over.use_premultiply = True
    alpha_over.premul = 1

    # Connect alpha over node
    render_layers = node_tree.nodes['Render Layers']
    composite = node_tree.nodes['Composite']
    node_tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
    node_tree.links.new(alpha_over.outputs['Image'], composite.inputs['Image'])

    # Render video
    scene.render.filepath = output_dir
    bpy.ops.render.render(animation=True)


def main():
    # Delete default cube
    set_mode('OBJECT')
    delete_all(obj_type='MESH')

    # Set lighting source emanating from camera
    camera_loc = bpy.data.objects['Camera'].location
    camera_rot = bpy.data.objects['Camera'].rotation_euler
    set_light_source('SUN', camera_loc, camera_rot)

    base_dir = 'objects'
    n_scenes = 64
    for scene_num in range(n_scenes, n_scenes+1):
        scene_dir = os.path.join(base_dir, 'scene_%03d' % scene_num)
        print('PROCESSING ', scene_dir.upper())
        image_dir = os.path.join(scene_dir, 'images')
        obj_file = os.path.join(scene_dir, 'textured.obj')
        data_file = os.path.join(scene_dir, 'data.npy')
        texture_file = os.path.join(scene_dir, 'texture.jpg')

        # Create random dot texture image and save to a file
        min_dot_diam = np.random.randint(5, 15)
        max_dot_diam = np.random.randint(20, 30)
        n_dots=np.random.randint(800, 1200)
        texture = dot_texture(min_diameter=min_dot_diam, max_diameter=max_dot_diam, n_dots=n_dots)
        texture.save(texture_file)

        # Load texture image and mesh
        img = load_img(texture_file)
        mesh = load_obj(obj_file)
        mesh_name = bpy.context.selected_objects[0].name

        # Wrap/edit texture
        wrap_texture(img)
        bsdf = bpy.data.materials[-1].node_tree.nodes['Principled BSDF']
        bsdf.inputs['Specular'].default_value = 0
        bsdf.inputs['Specular Tint'].default_value = 0
        # bsdf.inputs['Roughness'].default_value = 0
        # bsdf.inputs['Sheen Tint'].default_value = 0
        bsdf.inputs['Clearcoat'].default_value = 0

        # Animate rotation
        obj = bpy.data.objects[mesh_name]
        data = rotate(obj)

        # Save rotation parameters and mesh, and render images
        np.save(data_file, data)
        render(output_dir=image_dir)
        export_obj(mesh, scene_dir)

        delete_all(obj_type='MESH')

if __name__=='__main__':
    main()
