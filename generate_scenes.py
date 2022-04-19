import os

import bpy
import pdb
import sys
import time
import json
import copy
import bmesh
import pickle
import pathlib
import logging
import datetime
import numpy as np
from pprint import pprint
from pyquaternion import Quaternion
from mathutils.bvhtree import BVHTree

generator_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(generator_path)

import utils.utils as utils
import shapes
import configs
import textures
import BlenderArgparse
from rendering import RenderEngine

FORMAT = "%(asctime)-10s %(message)s"
print(os.getcwd())

parser = BlenderArgparse.ArgParser()
parser.add_argument(
    "--root_dir", type=str, help="Output directory for data", default="scenes"
)
parser.add_argument(
    "--n_scenes", type=int, help="Number of scenes to generate", default=1000
)
parser.add_argument(
    "--n_frames", type=int, help="Number of frames to render per scene", default=100
)
parser.add_argument(
    "--start_scene",
    type=int,
    help="Scene number to begin rendering from",
    default=0,
)
parser.add_argument(
    "--experiment_name", type=str, help="Experiment name", default="galaxy_scene_v1"
)
parser.add_argument(
    "--scene_config",
    type=str,
    help="path to config for scene",
    default="scene_config.json",
)
parser.add_argument("--save_config", action="store_true", help="save config")
parser.add_argument("--save_blendfile", action="store_true", help="save blendfile")
parser.add_argument(
    "--init_config_from_scene_dir",
    action="store_true",
    help="If present, will initialize config for individual scene directories",
)

parser.add_argument(
    "--no_texture_mesh",
    action="store_true",
    help="Whether or not to add dot texture to meshes",
)
# Render settings
parser.add_argument(
    "--render_views",
    type=str,
    default="textured",
    choices=["ground_truth", "textured", "all", "masks"],
    help="What types of views to render",
)

parser.add_argument("--device", type=str, help='Either "CUDA" or "CPU"', default="CUDA")

parser.add_argument(
    "--engine",
    type=str,
    help="rendering engine - either CYCLES or BLENDER_EEVEE",
    default="CYCLES",
)
parser.add_argument(
    "--output_img_name", type=str, help="Name for output images", default="Image"
)
parser.add_argument(
    "--render_size",
    type=int,
    nargs="+",
    help="size of .png file to render",
    default=512,
)
parser.add_argument(
    "--samples",
    type=int,
    default=64,
    help="Samples to use in rendering (fewer samples renders faster)",
)

# Background texture settings
parser.add_argument(
    "--background_style",
    type=str,
    default="textured",
    help="Options: white | textured | none",
)

parser.add_argument("--generate_sequential_textures", action="store_true")
parser.add_argument("--texture", type=str, default=None, help="Texture for scene")

args = parser.parse_args()


class BlenderScene(object):
    def __init__(
        self,
        scene_dir,
        scene_config=None,
        device="CUDA",
        engine="CYCLES",
        n_frames=100,
        render_size=512,
        samples=128,
        texture_scale=None,
        texture_distortion=None,
        *args,
        **kwargs,
    ):
        self.scene_dir = scene_dir
        self.data = bpy.data
        self.context = bpy.context
        self.n_frames = n_frames
        self.scene_config = scene_config
        self.renderer = RenderEngine(
            self.scene,
            device=device,
            engine=engine,
            render_size=render_size,
            samples=samples,
        )
        self.renderer.set_render_settings()

        self.rotation_data = os.path.join(self.scene_dir, "data.npy")
        self.render_frame_boundaries = None
        self.args = args
        self.kwargs = kwargs

        self.texture_scale = texture_scale
        self.texture_distortion = texture_distortion

        self.scene.frame_start = 1
        self.scene.frame_end = n_frames

        self.delete_all("MESH")

    @property
    def objects(self):
        return bpy.data.objects

    @property
    def scene(self):
        return bpy.data.scenes["Scene"]

    def delete_all(self, obj_type):
        """
        Deletes all instances of a given object type:
        obj_type: "MESH" | "LIGHT" | "CAMERA" ...
        """
        self.set_mode("OBJECT", obj_type)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.type == obj_type:
                self.delete(obj)

    def delete(self, obj):
        obj.select_set(True)
        bpy.ops.object.delete()

        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

    def set_mode(self, mode, obj_type="MESH"):
        for obj in self.objects:
            if obj.type == obj_type:
                self.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode=mode)

    def get_all_object_configs(self):
        object_config = self.scene_config.get("objects")
        if not object_config:
            raise ValueError(
                "Expected `objects` key in config, none found. \
                Please check to make sure your config is correct."
            )
        return object_config

    def intersection_check(self, obj, collection=None):
        # check every object for intersection with every other object
        self.set_mode("OBJECT")
        bpy.context.view_layer.update()

        def dist(loc1, loc2):
            loc1 = np.array(loc1)
            loc2 = np.array(loc2)
            return np.sqrt(np.sum((loc1 - loc2) ** 2))

        print(f"{obj.name}: {obj.location}")
        for col in self.data.collections:
            obj_list = col.all_objects
            for obj_next in obj_list:
                if obj == obj_next or obj_next.type != "MESH":
                    continue
                print(
                    f"\tChecking {obj.name} against: {obj_next.name}: {obj_next.location}"
                )
                obj_dist = dist(obj.location, obj_next.location)
                print("\tDISTANCE: ", obj_dist)
                if obj_dist < 4:
                    print(
                        f"{obj.name}: {obj.location} is too close to {obj_next.name}: {obj_next.location} -- intersection detected"
                    )
                    return True

        return False

    def add_objects(self, scene_config=None):
        """
        Generates all the shapes specified in scene config.
        If the BlenderScene was not initialized with a config,
        one must be specified when calling the function.
        """
        self.set_mode("OBJECT")
        all_object_configs = self.get_all_object_configs()

        # Add in each object from the config
        object_ids = all_object_configs.copy()
        for object_id in object_ids:
            object_config = all_object_configs[object_id]
            shape_params = object_config.get("shape_params")
            scaling_params = object_config.get("scaling_params")
            shape_type = object_config.get("shape_type")
            n_children = object_config.get("n_children")
            child_params = object_config.get("child_params")
            location = object_config.get("location")

            is_parent = True if child_params else False
            shape = shapes.create_shape(
                shape_type=shape_type,
                shape_params=shape_params,
                scaling_params=scaling_params,
                is_parent=is_parent,
                object_id=object_id,
            )
            print(shape)
            # Create meshes along the parent shape manifold
            # and update the config accordingly
            if is_parent:
                if "children" not in object_config.keys():
                    object_config["children"] = []

                child_shape_type = child_params.get("shape_type")
                child_shape_params = child_params.get("shape_params")
                child_scaling_params = child_params.get("scaling_params")

                i = 0
                location = location if location else np.array([0, 0, 0])

                verts = shape.sample_points(n_children, np.mean(scaling_params))
                verts += location
                for vert in verts:
                    if len(object_config["children"]) > i:
                        child_config = all_object_configs[object_config["children"][i]]
                        child_shape_type = child_config.get("shape_type")
                        child_shape_params = child_config.get("shape_params")
                        child_scaling_params = child_config.get("scaling_params")

                    child_id = f"{object_id}_{i}"
                    child_object = shapes.create_shape(
                        shape_type=child_shape_type,
                        shape_params=child_shape_params,
                        scaling_params=child_scaling_params,
                        object_id=child_id,
                        is_parent=False,
                        n_points=50,
                    )
                    child_object.obj_id = child_id
                    obj = child_object.add_mesh()
                    obj.location = vert
                    obj.keyframe_insert("location", frame=1)

                    bpy.context.view_layer.update()

                    print(f"Adding child ID: {child_id}")
                    all_object_configs[child_id] = {
                        "shape_type": child_object.shape_type,
                        "shape_params": child_object.shape_params,
                        "scaling_params": child_object.scaling_params,
                        "texture": child_params.get("texture"),
                        "rotation": child_params.get("rotation"),
                        "parent_id": object_id,
                    }
                    object_config["children"].append(child_id)

                    i += 1

            # Otherwise just add a mesh in normally and update the config to
            # reflect the shape parameters
            is_child = object_config.get("parent_id")
            if not is_parent and not is_child:
                shape.add_mesh()

                if location is None or location == "random":
                    loc_x = np.random.uniform(5, 12)
                    loc_y = np.random.uniform(-14, 6)
                    loc_z = np.random.uniform(0, 10)
                    location = (loc_x, loc_y, loc_z)

                self.objects[object_id].location = location
                self.objects[object_id].keyframe_insert("location", frame=1)

            object_config["shape_params"] = shape.shape_params
            object_config["shape_type"] = shape.shape_type
            object_config["scaling_params"] = shape.scaling_params

            print(all_object_configs.keys())

            return

    def add_background_plane(self, texture_params={}, add_displacement=True):
        """
        Adds plane to background and adds image texture that can be modified during animation.
        The texture material is added with the name "Background" for later access.
        """

        mesh = bpy.data.meshes.new("Plane")
        obj = bpy.data.objects.new("Plane", mesh)

        bpy.context.collection.objects.link(obj)

        bm = bmesh.new()
        bm.from_object(obj, bpy.context.view_layer.depsgraph)

        size = 100
        bm.verts.new((size, size, 0))
        bm.verts.new((size, -size, 0))
        bm.verts.new((-size, size, 0))
        bm.verts.new((-size, -size, 0))

        bmesh.ops.contextual_create(bm, geom=bm.verts)
        for f in bm.faces:
            f.select_set(True)
        bm.to_mesh(mesh)

        obj = self.data.objects["Plane"]
        self.set_mode("EDIT")
        status = bpy.ops.uv.unwrap()

        # Add texture to background plane
        textures.add_texture(self, obj, texture_params)

        if add_displacement:
            modifiers = obj.modifiers
            modifiers.new("Subdivision", type="SUBSURF")
            modifiers["Subdivision"].levels = 6
            modifiers["Subdivision"].render_levels = 6

            modifiers.new("Displacement", type="DISPLACE")
            self.data.textures.new("BackgroundDisplacement", "DISTORTED_NOISE")
            modifiers["Displacement"].texture = self.data.textures[
                "BackgroundDisplacement"
            ]
            strength = np.random.randint(5, 10)
            modifiers["Displacement"].strength = strength
            self.scene_config["background"]["displacement"] = strength
            self.set_mode("OBJECT")

        obj.rotation_euler = self.data.objects["Camera"].rotation_euler
        obj.location = [-15, 15, 0]

        return

    def set_background_color(self, color=(255, 255, 255, 1)):
        """
        color should be a tuple (R,G,B,alpha) where alpha controls
        the transparency. Default color is white
        """
        # Set transparent background for render
        scene = self.scene
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Raw"

        # Add alpha over node for colored background
        scene.use_nodes = True
        node_tree = scene.node_tree
        alpha_over = node_tree.nodes.new("CompositorNodeAlphaOver")

        alpha_over.inputs[1].default_value = color
        alpha_over.use_premultiply = True
        alpha_over.premul = 1

        # Connect alpha over node
        render_layers = node_tree.nodes["Render Layers"]
        composite = node_tree.nodes["Composite"]
        node_tree.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
        node_tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])

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
        utils.delete_all(obj_type="LIGHT")  # Delete existing lights
        light_data = bpy.data.lights.new(name="Light", type=light_type)
        light_data.energy = 5
        light_data.specular_factor = 0
        light_object = bpy.data.objects.new(name="Light", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object

        light_object.data.use_shadow = False
        light_object.cycles.cast_shadow = False

        light_object.location = location
        light_object.rotation_euler = rotation

    def rotate_object(self, object_id, object_config):
        n_frames = self.n_frames
        object_config["rotation_quaternion"] = np.zeros(shape=[n_frames, 4])
        object_config["rotation_matrix"] = np.zeros(shape=[n_frames, 3, 3])
        object_config["angle"] = np.zeros(shape=[n_frames, 1])
        object_config["axis"] = np.zeros(shape=[n_frames, 3])

        rotation_axis = np.random.uniform(-1, 1, 3)
        degrees = np.linspace(0, 360, n_frames)

        obj = self.data.objects[object_id]
        obj.rotation_mode = "QUATERNION"

        print(f"Rotation object {object_id}...")
        for frame, degree in enumerate(degrees):
            q = Quaternion(axis=rotation_axis, degrees=degree)

            # Record params
            object_config["rotation_quaternion"][frame] = q.elements
            object_config["rotation_matrix"][frame] = q.rotation_matrix
            object_config["angle"][frame] = degree
            object_config["axis"][frame] = rotation_axis

            # Animate rotation
            obj.rotation_quaternion = q.elements
            obj.keyframe_insert("rotation_quaternion", frame=frame + 1)

        print("...done.")
        return

    def rotate_hierarchy(self, hierarchy_config, collection_id, rotation="random"):
        """
        Updates the location of every child in hierarchy to respect rotated
        vertex location.
        Params:
            hierarchy_config: dict: contains dictionary of children and parameters for hierarchy
            collection_id: str: name of hierarchy (key to the hierarchy_config value)
            rotation: str: "random" | "noisy" | "none"
        """

        n_frames = self.n_frames
        object_config = self.get_all_object_configs()
        hierarchy_config["rotation_quaternion"] = np.zeros(shape=[n_frames, 4])
        hierarchy_config["rotation_matrix"] = np.zeros(shape=[n_frames, 3, 3])
        hierarchy_config["angle"] = np.zeros(shape=[n_frames, 1])
        hierarchy_config["axis"] = np.zeros(shape=[n_frames, 3])
        location = hierarchy_config.get("location", np.array([0, 0, 0]))

        # shape = shapes.create_shape(
        #     hierarchy_config["shape_type"],
        #     hierarchy_config["shape_params"],
        #     hierarchy_config["scaling_params"],
        #     is_parent=True,
        #     n_children=hierarchy_config["n_children"],
        # )

        verts = []
        for child_id in hierarchy_config["children"]:
            verts.append(self.objects[child_id].location)

        verts = np.array(verts)
        rotation_axis = np.random.uniform(-1, 1, 3)
        degrees = np.linspace(0, 360, n_frames)

        verts += location

        print(f"Rotating children in hierarchy: {collection_id}...")

        for frame, degree in enumerate(degrees):
            q = Quaternion(axis=rotation_axis, degrees=degree)
            rmat = q.rotation_matrix

            hierarchy_config["rotation_quaternion"][frame] = q.elements
            hierarchy_config["rotation_matrix"][frame] = rmat
            hierarchy_config["angle"][frame] = degree
            hierarchy_config["axis"][frame] = rotation_axis

            tmp_verts = np.matmul(verts, rmat)
            if rotation == "noisy":
                tmp_verts += np.random.uniform(-1, 1, size=tmp_verts.shape)

            for i, vert in enumerate(tmp_verts):
                child_id = f"{collection_id}_{i}"
                if child_id not in hierarchy_config["children"]:
                    continue

                child_config = object_config[child_id]
                if not isinstance(child_config.get("location"), np.ndarray):
                    child_config["location"] = np.zeros(shape=[n_frames, 3])

                child_config["location"][frame] = vert
                # Animate location change
                child_obj = self.data.objects[child_id]
                child_obj.location = vert
                child_obj.keyframe_insert("location", frame=frame + 1)

        print("...done.")
        return

    def generate_rotations(self, scene_config=None):
        if not scene_config:
            scene_config = self.scene_config

        all_object_configs = self.get_all_object_configs()
        for object_id in all_object_configs:
            object_config = all_object_configs[object_id]
            child_params = object_config.get("child_params")
            is_parent = True if child_params else False
            rotation = object_config.get("rotation")

            if rotation:
                if is_parent:
                    self.rotate_hierarchy(object_config, object_id, rotation)
                else:
                    self.rotate_object(object_id, object_config)

        return

    def add_textures(self, scene_config=None):
        all_object_configs = self.get_all_object_configs()
        for object_id in all_object_configs:
            object_config = all_object_configs[object_id]
            children = object_config.get("children")
            is_parent = True if children else False
            if is_parent:
                continue
            else:
                texture_config = object_config.get("texture")
                if self.texture_scale:
                    texture_config["params"]["Scale"] = self.texture_scale
                if self.texture_distortion:
                    distortion_key = self.texture_distortion[0]
                    texture_config["params"][distortion_key] = self.texture_distortion[
                        1
                    ]

                obj = self.data.objects[object_id]
                texture = textures.add_texture(self, obj, texture_config)
                object_config["texture"] = texture

    def generate_trajectory(self, obj, mesh_id=0, save=True):
        data = {}
        if os.path.exists(self.rotation_data):
            data = np.load(self.rotation_data, allow_pickle=True).item()

        if not data.get(mesh_id):
            object = {}

        n_frames = self.n_frames
        object["trajectory"] = np.zeros((n_frames, 3))

        quadrants = np.random.choice(
            list(self.render_frame_boundaries.keys()), 2, replace=False
        )
        locations = [self.render_frame_boundaries[quadrant] for quadrant in quadrants]
        print(quadrants)
        for i, quadrant in enumerate(quadrants):
            y, x = quadrant.split("_")
            if y == "top":
                y_range = locations[i][1] - self.y_len / 2
            else:
                y_range = locations[i][1] + self.y_len / 2

            if x == "right":
                x_range = locations[i][0] - self.x_len / 2
            else:
                x_range = locations[i][0] + self.x_len / 2

            # Randomly sample starting point along either x or y axis of a quadrant
            if np.random.rand() > 0.5:
                # Sample starting point along y region
                locations[i][1] = np.random.uniform(locations[i][1], y_range)
            else:
                locations[i][0] = np.random.uniform(locations[i][0], x_range)

        self.render_frame_boundaries.pop(quadrants[0])
        self.render_frame_boundaries.pop(quadrants[1])

        x1, y1, z1 = locations[0]
        x2, y2, z2 = locations[1]

        scene = self.scene
        scene.frame_start = 1
        scene.frame_end = self.n_frames

        obj.location = [x1, y1, z1]
        obj.keyframe_insert("location", frame=1)

        obj.location = [x2, y2, z2]
        obj.keyframe_insert("location", frame=self.n_frames)

        for i in range(n_frames):
            bpy.context.scene.frame_set(i)
            object["trajectory"][i] = obj.location

        bpy.context.scene.frame_set(1)

        if save:
            np.save(self.rotation_data, data, allow_pickle=True)
        return data

    def render(self, output_dir="images"):
        img_path = os.path.join(self.scene_dir, output_dir)
        self.renderer.render(img_path)

    def generate_masks(self, output_dir="masks"):
        objects = self.data.objects
        scene = self.context.scene

        scene.use_nodes = True

        # Give each object in the scene a unique pass index
        scene.view_layers["View Layer"].use_pass_object_index = True
        scene.view_layers["View Layer"].use_pass_normal = True
        scene.view_layers["View Layer"].use_pass_z = True
        scene.view_layers["View Layer"].use_pass_mist = True

        for i, object in enumerate(objects):
            if object.name == "Plane":
                object.pass_index = 0
            else:
                object.pass_index = (i + 1) * 20

        node_tree = scene.node_tree
        links = node_tree.links

        for node in node_tree.nodes:
            node_tree.nodes.remove(node)

        # Create a node for outputting the rendered image
        # image_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
        # image_output_node.name = "Image_Output"
        # image_output_node.label = "Image_Output"
        path = os.path.join(self.scene_dir, "images", "Image")
        self.scene.render.filepath = path
        # image_output_node.base_path = path
        # image_output_node.location = 600, 0

        # Create a node for outputting the depth of each pixel from the camera
        depth_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
        depth_output_node.name = "Depth_Output"
        depth_output_node.label = "Depth_Output"
        path = os.path.join(self.scene_dir, "depths")
        depth_output_node.base_path = path
        depth_output_node.location = 600, 0

        # Create a node for outputting shaded scenes
        """
        shaded_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
        shaded_output_node.label = "Shaded_Output"
        shaded_output_node.name = "Shaded_Output"
        path = os.path.join(self.scene_dir, "shaded")
        shaded_output_node.base_path = path
        shaded_output_node.location = 600, -100
        """
        # Create a node for outputting the index of each object
        mask_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
        mask_output_node.label = "Mask_Output"
        mask_output_node.name = "Mask_Output"
        mask_output_node.format.color_mode = "RGB"
        path = os.path.join(self.scene_dir, "masks")
        mask_output_node.base_path = path
        mask_output_node.location = 600, -200

        # Create a node for outputting the normal of each object
        normal_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
        normal_output_node.label = "Normal_Output"
        normal_output_node.name = "Normal_Output"
        normal_output_node.format.file_format = "OPEN_EXR"
        normal_output_node.format.color_mode = "RGB"
        path = os.path.join(self.scene_dir, "normals")
        normal_output_node.base_path = path
        normal_output_node.location = 600, -300

        math_node = node_tree.nodes.new(type="CompositorNodeMath")
        math_node.operation = "DIVIDE"
        math_node.inputs[1].default_value = 255.0
        math_node.location = 400, -200

        map_range_node = node_tree.nodes.new(type="CompositorNodeMapRange")
        normalize_node = node_tree.nodes.new(type="CompositorNodeNormalize")
        normalize_node.location = 100, 0
        map_range_node.location = 400, 0
        map_range_node.inputs["From Min"].default_value = 0
        map_range_node.inputs["From Max"].default_value = 1
        map_range_node.inputs["To Min"].default_value = 1
        map_range_node.inputs["To Max"].default_value = 0

        # Create a node for the output from the renderer
        compositor_node = node_tree.nodes.new(type="CompositorNodeComposite")
        compositor_node.location = 600, 200
        render_layers_node = node_tree.nodes.new(type="CompositorNodeRLayers")
        render_layers_node.location = -100, 0

        # Link all the nodes together
        links.new(render_layers_node.outputs["Image"], compositor_node.inputs["Image"])

        # Link Depth
        links.new(render_layers_node.outputs["Depth"], normalize_node.inputs[0])
        links.new(normalize_node.outputs[0], map_range_node.inputs[0])
        links.new(map_range_node.outputs[0], depth_output_node.inputs["Image"])

        # Link Object Index Masks
        links.new(render_layers_node.outputs["IndexOB"], math_node.inputs[0])
        links.new(math_node.outputs[0], mask_output_node.inputs["Image"])
        """
        # Link Shaded
        links.new(
            render_layers_node.outputs["Normal"], shaded_output_node.inputs["Image"]
        )
        """
        # Link Normals
        links.new(
            render_layers_node.outputs["Normal"], normal_output_node.inputs["Image"]
        )

    def generate_ground_truth(self, output_dir="images"):
        output_dir = os.path.join(self.scene_dir, output_dir, "Image")
        self.scene.render.filepath = output_dir
        background_plane = bpy.data.objects["Plane"]
        self.set_mode("OBJECT", "MESH")
        self.delete(background_plane)

        world_nodes = self.data.worlds["World"].node_tree.nodes
        world_nodes["Background"].inputs["Color"].default_value = np.random.rand(4)
        world_nodes["Background"].inputs["Strength"].default_value = 1
        self.objects["Light"].data.energy = 1.5
        self.objects["Light"].data.use_shadow = True

        print("[Turning off texture]")
        for mat in bpy.data.materials:
            mat.use_nodes = False
            print(mat.name)

        for obj in bpy.data.objects:
            if obj.type == "MESH":
                obj.active_material.diffuse_color = list(np.random.rand(3)) + [1]
                obj.active_material.shadow_method = "OPAQUE"

        self.context.scene.render.film_transparent = False

        self.renderer.set_render_settings()

    def create_default_scene(self, args):
        # Clear any previous meshes
        self.set_mode("OBJECT")
        self.delete_all(obj_type="MESH")
        self.delete_all(obj_type="OBJECT")

        self.delete_all(obj_type="LIGHT")

        # Set direct and ambient light
        camera_loc = bpy.data.objects["Camera"].location
        camera_rot = bpy.data.objects["Camera"].rotation_euler
        self.set_light_source("SUN", camera_loc, camera_rot)

        self.add_objects()
        self.generate_rotations()
        self.add_textures()

        # if args.scene_type == "galaxy":
        #     bpy.ops.object.empty_add(
        #         type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        #     )
        #     center_axis = bpy.context.active_object
        #     self.generate_rotation(center_axis, mesh_id=-1)

        background = self.scene_config.get("background")
        if not background:
            # Set background to white
            world_nodes = self.data.worlds["World"].node_tree.nodes
            world_nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
            world_nodes["Background"].inputs["Strength"].default_value = 1.5
            self.set_background_color(color=(255, 255, 255, 1))
        else:
            texture = background.get("texture")
            if self.texture_scale:
                texture["params"]["Scale"] = self.texture_scale
            if self.texture_distortion:
                distortion_key = self.texture_distortion[0]
                texture["params"][distortion_key] = self.texture_distortion[1]

            displacement = background.get("displacement")
            self.add_background_plane(
                texture_params=texture, add_displacement=displacement
            )

        if args.render_views == "ground_truth":
            self.generate_masks()
            self.generate_ground_truth()
            bpy.ops.render.render(animation=True, write_still=True)
        elif args.render_views == "masks":
            self.generate_masks()
            bpy.ops.render.render(animation=True, write_still=True)
        else:
            print(f"{args.render_views} is not defined. No images will be rendered.")

        # Save and clean up scene
        if args.save_blendfile:
            bpy.ops.wm.save_mainfile(
                filepath=self.scene_dir + "/scene.blend", check_existing=False
            )
        if args.save_config:
            self.save_config()
        self.delete_all(obj_type="MESH")

        return

    def save_config(self):
        config_path = os.path.join(self.scene_dir, "scene_config.pkl")
        print(f"Writing config to {config_path}")
        with open(config_path, "wb") as f:
            pickle.dump(self.scene_config, f, protocol=pickle.HIGHEST_PROTOCOL)


def sequential_texture_gen(args):
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir, exist_ok=True)
        print("Created root directory: ", args.root_dir)

    texture_maps = textures.TEXTURE_MAPS
    texture = textures.Shader(args.texture)
    texture_params = texture_maps[texture]
    scale_range = texture_params["Scale"]
    if args.texture == "voronoi":
        distortion_opt = "Randomness"
    else:
        distortion_opt = "Distortion"
    distortion_range = texture_params[distortion_opt]

    # Option to match scale and distortion bins by taking sqrt of
    # desired scenes. For now hard code ~50 bins total
    n_scale_bins = 3
    n_distortion_bins = 3
    scenes_per_bin = 10
    scale_bins = np.linspace(scale_range[0], scale_range[1], n_scale_bins)
    distortion_bins = np.linspace(distortion_range[0], distortion_range[1], n_distortion_bins)

    scene_num = args.start_scene
    for scale in scale_bins:
        for distortion in distortion_bins:
            for n in range(scenes_per_bin):
                scene_dir = os.path.join(args.root_dir, f"scale={scale:.3f}_distortion={distortion:.3f}_scene={n}")
                os.makedirs(scene_dir, exist_ok=True)

                if args.scene_config != "random":
                    if args.init_config_from_scene_dir:
                        config_path = os.path.join(scene_dir, args.scene_config)
                    else:
                        config_path = args.scene_config
                    print("Loading config from path: ", config_path)
                    with open(config_path, "rb") as f:
                        if config_path.endswith(".pkl"):
                            scene_config = pickle.load(f)
                        else:
                            scene_config = json.load(f)
                            pprint(scene_config)
                else:
                    scene_config = configs.generate_random_config()
                    print("Generated random scene config: ")
                    pprint(scene_config)

                # Create a scene and initialize some basic properties
                scene = BlenderScene(
                    scene_dir,
                    scene_config=scene_config,
                    render_size=args.render_size,
                    device=args.device,
                    n_frames=args.n_frames,
                    engine=args.engine,
                    samples=args.samples,
                    texture_scale=scale,
                    texture_distortion=(distortion_opt, distortion),
                )
                scene.n_frames = args.n_frames

                # Set camera properties for scene
                camera = scene.data.objects["Camera"]
                camera.location = [
                    18.554821014404297,
                    -18.291574478149414,
                    12.243793487548828,
                ]
                camera.rotation_euler = [
                    1.1093190908432007,
                    9.305318826591247e-08,
                    0.8149283528327942,
                ]
                camera.data.sensor_width = 50

                scene.create_default_scene(args)

                # Add a render timestamp
                ts = time.time()
                fts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                with open(os.path.join(scene_dir, "timestamp.txt"), "w") as f:
                    f.write("Files Rendered at: {}\n".format(fts))

                scene_num += 1


def main(args):
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir, exist_ok=True)
        print("Created root directory: ", args.root_dir)

    for scene_num in range(args.start_scene, args.start_scene + args.n_scenes):
        scene_dir = os.path.join(args.root_dir, "scene_%03d" % scene_num)
        os.makedirs(scene_dir, exist_ok=True)

        if args.scene_config != "random":
            if args.init_config_from_scene_dir:
                config_path = os.path.join(scene_dir, args.scene_config)
            else:
                config_path = args.scene_config
            print("Loading config from path: ", config_path)
            with open(config_path, "rb") as f:
                if config_path.endswith(".pkl"):
                    scene_config = pickle.load(f)
                else:
                    scene_config = json.load(f)
                    pprint(scene_config)
        else:
            scene_config = configs.generate_random_config()
            print("Generated random scene config: ")
            pprint(scene_config)

        # Create a scene and initialize some basic properties
        scene = BlenderScene(
            scene_dir,
            scene_config=scene_config,
            render_size=args.render_size,
            device=args.device,
            n_frames=args.n_frames,
            engine=args.engine,
            samples=args.samples,
        )
        scene.n_frames = args.n_frames

        # Set camera properties for scene
        camera = scene.data.objects["Camera"]
        camera.location = [18.554821014404297, -18.291574478149414, 12.243793487548828]
        camera.rotation_euler = [
            1.1093190908432007,
            9.305318826591247e-08,
            0.8149283528327942,
        ]
        camera.data.sensor_width = 50

        scene.create_default_scene(args)

        # Add a render timestamp
        ts = time.time()
        fts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(scene_dir, "timestamp.txt"), "w") as f:
            f.write("Files Rendered at: {}\n".format(fts))


if __name__ == "__main__":
    if args.generate_sequential_textures:
        print("Generating sequential textured scenes")
        sequential_texture_gen(args)
    else:
        main(args)
