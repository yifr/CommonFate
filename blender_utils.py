import os
import bpy

def set_mode(mode, type='MESH'):
    """
    Sets the mode for a specific object type
    
    Parameters
    -----------
        mode: :str: : Blender mode to set ('EDIT' | 'OBJECT' | etc...)
        type: :str: : Object type to apply setting to
    """
    scene = bpy.context.scene

    for obj in scene.objects:
        if obj.type == type:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode=mode)

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

def cube_project():
    set_mode('EDIT')
    bpy.ops.uv.cube_project(cube_size=1)

def load_img(path):
    img = bpy.data.images.load(filepath=path)
    return img

def load_obj(path):
    mesh = bpy.ops.import_scene.obj(filepath=path)
    return mesh

def export_obj(obj, scene_dir, fname='mesh.obj'):
    """
    Exports textured mesh to a file called textured.obj.
    Also exports material group used to texture
    Assumes one active mesh at a time
    """
    set_mode('OBJECT')
    output_file = os.path.join(scene_dir, fname)
    bpy.ops.export_scene.obj(filepath=output_file, use_selection=True)