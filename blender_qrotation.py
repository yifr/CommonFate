import os
import bpy
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion


def set_mode(mode):
    scene = bpy.context.scene
    #scene.layers = [True] * 20 # Show all layers

    for obj in scene.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode=mode)


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

def delete_all(type):
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
        if o.type == type:
            o.select_set(True)
        else:
            o.select_set(False)
    set_mode('OBJECT')
    bpy.ops.object.delete()
    
def set_light_source(type, location, rotation):
    """
    Creates a single light source at a specific location / rotation
    
    Parameters
    -----------
        type: Light type ('POINT' | 'SUN' | 'SPOT' | 'AREA')
        location: Vector location of light
        rotation: Euler angle rotation of light
    """    
    type = type.upper()
    delete_all(type='LIGHT') # Delete existing lights
    light_data = bpy.data.lights.new(name='Light', type=type)
    light_object = bpy.data.objects.new(name='Light', object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    bpy.context.view_layer.objects.active = light_object
    
    light_object.location = location
    light_object.rotation_euler = rotation
    
    
set_mode('EDIT')

# Set lighting source emanating from camera
camera_loc = bpy.data.objects['Camera'].location
camera_rot = bpy.data.objects['Camera'].rotation_euler
set_light_source('Sun', camera_loc, camera_rot)

obj = bpy.data.objects['mesh.118']

# Remove specular tint
mat_nodes = bpy.data.materials[-1].node_tree.nodes
mat_nodes['Principled BSDF'].inputs[5].default_value = 0

data = rotate(obj)
np.save('/Users/yoni/MIT/projects/CommonFate/btest/data.npy', data)
render('/Users/yoni/MIT/projects/CommonFate/btest/images')