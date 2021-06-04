import os
import bpy


def set_mode(mode, obj_type="MESH"):
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
    set_mode("OBJECT")
    bpy.ops.object.delete()


def add_mesh(name, verts, faces, edges=None, col_name="Collection"):
    """
    Adds a mesh with a given name, defined vertices and faces to a collection
    """
    if edges is None:
        edges = []
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)


def cube_project():
    set_mode("EDIT")
    bpy.ops.uv.cube_project(cube_size=1)


def load_img(path):
    img = bpy.data.images.load(filepath=path)
    return img


def load_obj(path):
    mesh = bpy.ops.import_scene.obj(filepath=path)
    return mesh


def export_obj(obj, scene_dir, fname="textured.obj"):
    """
    Exports textured mesh to a file called textured.obj.
    Also exports material group used to texture
    Assumes one active mesh at a time
    """
    set_mode("OBJECT")
    output_file = os.path.join(scene_dir, fname)
    bpy.ops.export_scene.obj(filepath=output_file, use_selection=True)
