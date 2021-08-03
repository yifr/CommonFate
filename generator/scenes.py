import os

from numpy.core.numeric import full
import bpy
import pdb
import sys
import time
import json
import copy
import bmesh
import pathlib
import pickle
import logging
import datetime
import numpy as np
from pyquaternion import Quaternion
from mathutils.bvhtree import BVHTree

generator_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(generator_path)

import utils
import shapes
import BlenderArgparse
import textures
from rendering import RenderEngine

FORMAT = "%(asctime)-10s %(message)s"
logging.basicConfig(filename="render_logs", level=logging.INFO, format=FORMAT)
print(os.getcwd())

parser = BlenderArgparse.ArgParser()
parser.add_argument(
    "--root_dir", type=str, help="Output directory for data", default="scenes"
)
parser.add_argument(
    "--n_scenes", type=int, help="Number of scenes to generate", default=1000
)
parser.add_argument(
    "--n_frames", type=int, help="Number of frames to render per scene", default=10
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
parser.add_argument(
    "--no_texture_mesh",
    action="store_true",
    help="Whether or not to add dot texture to meshes",
)
# Render settings
parser.add_argument("--device", type=str, help='Either "CUDA" or "CPU"', default="CUDA")
parser.add_argument(
    "--engine",
    type=str,
    help="rendering engine - either CYCLES or BLENDER_EEVEE",
    default="CYCLES",
)
parser.add_argument(
    "--output_img_name", type=str, help="Name for output images", default="img"
)
parser.add_argument(
    "--render_size", type=int, help="size of .png file to render", default=512
)
parser.add_argument(
    "--samples",
    type=int,
    default=256,
    help="Samples to use in rendering (fewer samples renders faster",
)

# Background texture settings
parser.add_argument(
    "--background_style",
    type=str,
    default="textured",
    help="Options: white | textured | none",
)
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
        self.rotation_data = os.path.join(self.scene_dir, "data.npy")
        self.render_frame_boundaries = None
        self.args = args
        self.kwargs = kwargs

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

    def get_object_config(self):
        object_config = self.scene_config.get("objects")
        if not object_config:
            raise ValueError(
                "Expected `objects` key in config, none found. \
                Please check to make sure your config is correct."
            )
        return object_config

    def bmesh_copy_from_object(
        self, obj, transform=True, triangulate=True, apply_modifiers=False
    ):
        """
        Returns a transformed, triangulated copy of the mesh
        """

        assert obj.type == "MESH"

        if apply_modifiers and obj.modifiers:
            me = obj.to_mesh(bpy.context.scene, True, "PREVIEW", calc_tessface=False)
            bm = bmesh.new()
            bm.from_mesh(me)
            bpy.data.meshes.remove(me)
        else:
            me = obj.data
            if obj.mode == "EDIT":
                bm_orig = bmesh.from_edit_mesh(me)
                bm = bm_orig.copy()
            else:
                bm = bmesh.new()
                bm.from_mesh(me)

        # Remove custom data layers to save memory
        for elem in (bm.faces, bm.edges, bm.verts, bm.loops):
            for layers_name in dir(elem.layers):
                if not layers_name.startswith("_"):
                    layers = getattr(elem.layers, layers_name)
                    for layer_name, layer in layers.items():
                        layers.remove(layer)

        if transform:
            bm.transform(obj.matrix_world)

        if triangulate:
            bmesh.ops.triangulate(bm, faces=bm.faces)

        return bm

    def bmesh_check_intersect_objects(self, obj, obj2):
        """
        Check if any faces intersect with the other object

        returns a boolean
        """
        assert obj != obj2

        # Triangulate
        bm = self.bmesh_copy_from_object(obj, transform=True, triangulate=True)
        bm2 = self.bmesh_copy_from_object(obj2, transform=True, triangulate=True)

        # If bm has more edges, use bm2 instead for looping over its edges
        # (so we cast less rays from the simpler object to the more complex object)
        if len(bm.edges) > len(bm2.edges):
            bm2, bm = bm, bm2

        # Create a real mesh (lame!)
        scene = bpy.context.scene
        me_tmp = bpy.data.meshes.new(name="~temp~")
        bm2.to_mesh(me_tmp)
        bm2.free()
        obj_tmp = bpy.data.objects.new(name=me_tmp.name, object_data=me_tmp)
        bpy.context.collection.objects.link(obj_tmp)
        bpy.context.view_layer.update()
        ray_cast = obj_tmp.ray_cast

        intersect = False

        EPS_NORMAL = 0.000001
        EPS_CENTER = 0.01  # should always be bigger

        # for ed in me_tmp.edges:
        for ed in bm.edges:
            v1, v2 = ed.verts

            # setup the edge with an offset
            co_1 = v1.co.copy()
            co_2 = v2.co.copy()
            co_mid = (co_1 + co_2) * 0.5
            no_mid = (v1.normal + v2.normal).normalized() * EPS_NORMAL
            co_1 = co_1.lerp(co_mid, EPS_CENTER) + no_mid
            co_2 = co_2.lerp(co_mid, EPS_CENTER) + no_mid

            success, co, no, index = ray_cast(co_1, co_2 - co_1, ed.calc_length())
            if success:
                intersect = True
                break

        bpy.context.collection.objects.unlink(obj_tmp)
        bpy.data.objects.remove(obj_tmp)
        bpy.data.meshes.remove(me_tmp)

        bpy.context.view_layer.update()

        return intersect

    def intersection_check(self, obj, collection=None):
        # check every object for intersection with every other object
        self.set_mode("OBJECT")
        bpy.context.view_layer.update()

        for col in self.data.collections:
            obj_list = col.all_objects
            for obj_next in obj_list:
                if obj == obj_next or obj_next.type != "MESH":
                    print(obj, obj_next, "continuing")
                    continue

                def dist(loc1, loc2):
                    loc1 = np.array(loc1)
                    loc2 = np.array(loc2)
                    return np.sqrt(np.sum((loc1 - loc2) ** 2))

                obj_dist = dist(
                    obj.matrix_world.translation, obj_next.matrix_world.translation
                )

                if obj_dist < 0.05:
                    print("Objects are too close -- intersection detected")
                    return True

                # bm1 = self.bmesh_copy_from_object(obj)
                # bm2 = self.bmesh_copy_from_object(obj_next)

                # bm1.transform(obj.matrix_world)
                # bm2.transform(obj_next.matrix_world)

                # # make BVH tree from BMesh of objects
                # obj_now_BVHtree = BVHTree.FromBMesh(bm1)
                # obj_next_BVHtree = BVHTree.FromBMesh(bm2)

                # # get intersecting pairs
                # inter = obj_now_BVHtree.overlap(obj_next_BVHtree)
                # if len(inter) > 10:
                #     return True

        print("No intersections detected")
        return False

    def add_mesh(self, id, verts, faces, collection=None):
        """
        Adds a mesh to Blender with specified id and collection (if specified)
        Params:
            id: str: name of mesh
            verts: list: list of vertices
            faces: list: list of faces
        """
        mesh = self.data.meshes.new(id)
        obj = self.data.objects.new(mesh.name, mesh)

        # Objects are assigned to unique collections if they belong
        # to a larger shape hierarchy
        if collection:
            col = self.data.collections.get(collection)
            if not col:
                col = self.data.collections.new(collection)
                self.context.scene.collection.children.link(col)
        else:
            col = self.data.collections.get("Collection")

        col.objects.link(obj)
        # self.context.view_layer.objects.active = obj
        mesh.from_pydata(verts, [], faces)
        return obj

    def add_objects(self, scene_config=None):
        """
        Generates all the shapes specified in scene config.
        If the BlenderScene was not initialized with a config,
        one must be specified when calling the function.
        """
        self.set_mode("OBJECT")
        full_object_config = self.get_object_config()

        # Add in each object from the config
        object_ids = full_object_config.copy()
        for object_id in object_ids:
            object_config = full_object_config[object_id]
            shape_params = object_config.get("shape_params")
            scaling_params = object_config.get("scaling_params")
            shape_type = object_config.get("shape_type")
            child_params = object_config.get("child_params")
            n_points = object_config.get("n_points", 50)
            location = object_config.get("location")

            is_parent = True if child_params else False
            object = shapes.create_shape(
                shape_type,
                shape_params,
                scaling_params,
                is_parent=is_parent,
                n_points=n_points,
            )
            faces = object.faces
            verts = object.verts

            # Create meshes along the parent shape manifold
            # and update the config accordingly
            if is_parent:
                object_config["children"] = []
                child_shape_type = child_params.get("shape_type")
                child_shape_params = child_params.get("shape_params")
                child_scaling_params = child_params.get("scaling_params")

                # intersection_size = child_scaling_params[-1]
                i = 0
                for vert in verts:

                    child_object = shapes.create_shape(
                        child_shape_type,
                        child_shape_params,
                        child_scaling_params,
                        False,
                        n_points,
                    )
                    child_id = f"{object_id}_{i}"

                    child_verts = child_object.verts
                    child_faces = child_object.faces
                    obj = self.add_mesh(
                        child_id, child_verts, child_faces, collection=object_id
                    )

                    obj.location = vert
                    obj.keyframe_insert("location", frame=1)

                    bpy.context.view_layer.update()
                    intersection = self.intersection_check(obj, object_id)
                    if intersection:
                        self.delete(obj)
                    else:
                        print(f"Adding child ID: {child_id}")
                        print(obj.location)
                        full_object_config[child_id] = {
                            "shape_type": child_object.shape_type,
                            "shape_params": child_object.shape_params,
                            "scaling_params": child_object.scaling_params,
                            "texture": child_params.get("texture"),
                            "rotation": child_params.get("rotation"),
                        }
                        object_config["children"].append(child_id)

                        i += 1

            # Otherwise just add a mesh in normally and update the config to
            # reflect the shape parameters
            if not is_parent:
                object_config["shape_params"] = object.shape_params
                object_config["shape_type"] = object.shape_type
                object_config["scaling_params"] = object.scaling_params

                self.add_mesh(object_id, verts, faces)
                if location:
                    if location == "random":
                        location = np.random.uniform(-10, 10, 3)
                    self.objects[object_id].location = location
                    self.objects[object_id].keyframe_insert("location", frame=1)

        return

    def add_background_plane(self, texture_params={}):
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
        textures.add_texture(texture_params)

        obj.rotation_euler = [np.radians(91), np.radians(0), np.radians(45)]
        obj.location = [-10, 10, 0]

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
        light_object.cycles["cast_shadow"] = False

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
        hierarchy_config["rotation_quaternion"] = np.zeros(shape=[n_frames, 4])
        hierarchy_config["rotation_matrix"] = np.zeros(shape=[n_frames, 3, 3])
        hierarchy_config["angle"] = np.zeros(shape=[n_frames, 1])
        hierarchy_config["axis"] = np.zeros(shape=[n_frames, 3])

        shape = shapes.create_shape(
            hierarchy_config["shape_type"],
            hierarchy_config["shape_params"],
            hierarchy_config["scaling_params"],
            is_parent=True,
            n_points=hierarchy_config["n_points"],
        )

        rotation_axis = np.random.uniform(-1, 1, 3)
        degrees = np.linspace(0, 360, n_frames)
        verts = shape.verts

        print(f"Rotating children in hierarchy: {collection_id}...")
        print([x for x in self.data.objects])

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
                if not hierarchy_config["children"].get(child_id):
                    continue

                child_config = hierarchy_config["children"][child_id]
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

        full_object_config = self.get_object_config()
        for object_id in full_object_config:
            object_config = full_object_config[object_id]
            children = object_config.get("children")
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
        full_object_config = self.get_object_config()
        for object_id in full_object_config:
            object_config = full_object_config[object_id]
            children = object_config.get("children")
            is_parent = True if children else False
            if is_parent:
                child_config = object_config.get("child_params")
                child_texture = child_config.get("texture")
                for child_id in object_config["children"]:
                    obj = self.data.objects[child_id]
                    textures.add_texture(self, obj, child_texture)
            else:
                texture_config = object_config.get("texture")
                obj = self.data.objects[object_id]
                textures.add_texture(self, obj, texture_config)

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

        if args.background_style == "white":
            world_nodes = self.data.worlds["World"].node_tree.nodes
            world_nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
            world_nodes["Background"].inputs["Strength"].default_value = 1.5
            self.set_background_color(color=(255, 255, 255, 1))

        # if args.scene_type == "galaxy":
        #     bpy.ops.object.empty_add(
        #         type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        #     )
        #     center_axis = bpy.context.active_object
        #     self.generate_rotation(center_axis, mesh_id=-1)

        if args.background_style == "textured":
            self.add_background_plane(
                texture_params={"noise": args.background_noise},
                overwrite=args.new_background,
            )

        # for obj in self.objects:
        #     obj.select_set(True)

        bpy.ops.wm.save_as_mainfile(
            filepath=self.scene_dir + "/scene.blend", check_existing=False
        )
        self.render()

        # Clean up scene
        self.delete_all(obj_type="MESH")
        self.save_config()

        return

    def save_config(self):
        config_path = os.path.join(self.scene_dir, "scene_config.pkl")
        print(f"Writing config to {config_path}")
        with open(config_path, "wb") as f:
            pickle.dump(self.scene_config, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(args):
    if not os.path.exists(args.root_dir):
        os.mkdir(args.root_dir)
        print("Created root directory: ", args.root_dir)

    with open(args.scene_config, "r") as f:
        scene_config = json.load(f)

    for scene_num in range(args.start_scene, args.start_scene + args.n_scenes):
        scene_dir = os.path.join(args.root_dir, "scene_%03d" % scene_num)
        os.makedirs(scene_dir, exist_ok=True)
        logging.info("Processing scene: {}...".format(scene_dir))

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
        # scene.render_frame_boundaries = copy.deepcopy(
        #     global_scene_params["render_frame_boundaries"]
        # )
        # scene.x_len = (
        #     scene.render_frame_boundaries["top_right"][0]
        #     - scene.render_frame_boundaries["top_left"][0]
        # )
        # scene.y_len = (
        #     scene.render_frame_boundaries["top_right"][1]
        #     - scene.render_frame_boundaries["bottom_right"][1]
        # )

        scene.create_default_scene(args)

        # Add a render timestamp
        ts = time.time()
        fts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(scene_dir, "timestamp.txt"), "w") as f:
            f.write("Files Rendered at: {}\n".format(fts))


if __name__ == "__main__":
    main(args)
