import os
import bpy
import sys
import numpy as np
import pathlib
from glob import glob
import pickle
import hashlib

generator_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(os.path.join(generator_path, ".."))
import shapes
import utils.utils as utils


def add_mesh(name, verts, faces, edges=None, col_name="Collection"):
    if edges is None:
        edges = []
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get(col_name)
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts, edges, faces)

def load_single_shape(shape_type, shape_params, color):
    utils.delete_all("MESH")
    scaling = [2, 2, 2] if shape_type == "superellipsoid" else [1, 1, 1, 2]
    temp = shapes.SuperQuadric(shape_type, shape_params, scaling)
    add_mesh("superquadric", temp.verts, temp.faces)
    material = bpy.data.materials.new(name="Material")
    material.use_nodes = True

    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color

    ob = bpy.context.active_object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = material
    else:
        # no slots
        ob.data.materials.append(material)


def set_render_settings(render_size, samples=128):
    # Set properties to increase speed of render time
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"  # use cycles for headless rendering

    scene.render.resolution_x = render_size[0]
    scene.render.resolution_y = render_size[1]

    scene.render.image_settings.compression = 50
    scene.cycles.samples = samples


def enable_gpus(device_type="CUDA", use_cpus=False):
    """
    Sets device as GPU and adjusts rendering tile size accordingly
    """
    scene = bpy.context.scene
    scene.cycles.device = "GPU"

    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.compute_device_type = device_type

    activated_gpus = []
    print(cycles_preferences.get_devices())
    for device in cycles_preferences.devices:
        print("Activating: ", device)
        if not device.type == "CPU":
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type

    scene.render.tile_x = 128
    scene.render.tile_y = 128
    return activated_gpus


def get_all_shape_params(root, subdirs):
    shape_params = dict()
    for subdir in subdirs:
        scenes = glob(os.path.join(root, subdir, "*", "scene_*"))
        for idx, scene in enumerate(scenes):
            if idx > 200:
                break

            scene_config = os.path.join(scene, "scene_config.pkl")
            config = pickle.load(open(scene_config, "rb"))["objects"]
            objects = list(config.keys())
            for i in range(len(objects) - 1):
                obj_key = f"h1_{i}"
                obj_config = config[obj_key]
                params = tuple(obj_config["shape_params"])
                shape_type = obj_config["shape_type"]
                print(params[0], params[1], shape_type)
                shape_params[(params[0], params[1])] = shape_type

    return shape_params

def render_shapes(scene_shape_params):
    set_render_settings((512, 512))
    enable_gpus()
    color = list(np.random.uniform(0, 1, 4))
    for shape, params in scene_shape_params.items():
        shape = np.random.choice(["supertoroid", "superellipsoid"])
        params = np.random.uniform(0.1, 4, 2)

        filekey = f"{shape}_{params[0]}_{params[1]}".encode("utf-8")
        hexobj = hashlib.sha256(filekey)
        hexcode = hexobj.hexdigest()
        print(hexcode)
        bpy.context.scene.render.filepath = f"/om/user/yyf/CommonFate/media/2-afc/{hexcode}.png"
        bpy.context.scene.render.engine = "CYCLES"

        load_single_shape(shape, params, color)
        bpy.context.object.rotation_euler = [0.05, 0.05, 0.1]
        bpy.ops.render.render(write_still=True)


shape_params = get_all_shape_params("/om/user/yyf/CommonFate/scenes", ["test_voronoi", "test_wave", "test_noise"])
render_shapes(shape_params)
