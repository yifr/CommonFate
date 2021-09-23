import os
import bpy
import sys
import time
import json
import copy
import bmesh
import pathlib
import logging
import datetime
import numpy as np
from pyquaternion import Quaternion

generator_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(generator_path)

import utils
import shapes
import BlenderArgparse
import textures
import RenderEngine

FORMAT = "%(asctime)-10s %(message)s"
logging.basicConfig(filename="render_logs", level=logging.INFO, format=FORMAT)
print(os.getcwd())


ROOT_SCENE_DIR = "/om2/user/yyf/CommonFate/data/"


class BlenderScene(object):
    def __init__(
        self,
        scene_dir,
        shape_params=None,
        device="CUDA",
        engine="CYCLES",
        n_frames=100,
        render_size=512,
        samples=128,
    ):
        self.scene_dir = scene_dir
        self.data = bpy.data
        self.context = bpy.context
        self.n_frames = n_frames
        self.shape_params = shape_params
        self.renderer = RenderEngine(
            self.scene,
            device=device,
            engine=engine,
            render_size=render_size,
            samples=samples,
        )
        self.rotation_data = os.path.join(self.scene_dir, "data.npy")
        self.render_frame_boundaries = None
        self.delete_all("MESH")

    @property
    def objects(self):
        return bpy.data.objects

    @property
    def scene(self):
        return bpy.data.scenes["Scene"]

    def get_shape_params(self):
        if self.shape_params:
            return self.shape_params

        param_file = os.path.join(self.scene_dir, "params.json")
        if not os.path.exists(param_file):
            return None

        with open(param_file, "r") as f:
            params = json.load(f)
            self.shape_params = params

        return params

    def delete_all(self, obj_type):
        """
        Deletes all instances of a given object type:
        obj_type: "MESH" | "LIGHT" | "CAMERA" ...
        """
        for obj in self.objects:
            if obj.type == obj_type:
                obj.select_set(True)
            else:
                obj.select_set(False)
        self.set_mode("OBJECT", obj_type)
        bpy.ops.object.delete()

    def set_mode(self, mode, obj_type="MESH"):
        for obj in self.objects:
            if obj.type == obj_type:
                self.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode=mode)

    def _generate_mesh(self, shape_params=None, save=False):
        """
        shape_params: json dict with keys: 'scaling' and 'exponents'
        Returns points for a superquadric given some shape parameters
        """
        if self.get_shape_params() is None and shape_params is None:
            exponents = list(np.random.uniform(0, 4, 2))
            scaling = [1, 1, 1, 2]
            shape_params = {"mesh_0": {"scaling": scaling, "exponents": exponents}}
            self.shape_params = shape_params
            # Save parameters
            if save:
                self.shape_param_file = os.path.join(self.scene_dir, "params.json")
                with open(self.shape_param_file, "w") as f:
                    json.dump(self.shape_params, f)

        n_points = 100
        x, y, z = shapes.superellipsoid(
            shape_params["mesh_0"]["exponents"],
            shape_params["mesh_0"]["scaling"],
            n_points,
        )

        return x, y, z

    def create_mesh(self, shape_params=None):
        """
        Adds a mesh from shape parameters
        """
        if shape_params:
            self.shape_params = shape_params

        x, y, z = self._generate_mesh(shape_params=shape_params)
        faces, verts = shapes.get_faces_and_verts(x, y, z)
        edges = []

        mesh = self.data.meshes.new("mesh")
        obj = self.data.objects.new(mesh.name, mesh)
        col = self.data.collections.get("Collection")
        col.objects.link(obj)

        self.context.view_layer.objects.active = obj
        mesh.from_pydata(verts, edges, faces)

        return obj

    def load_mesh(self, mesh_id=0, save=False):
        old_objs = set(self.objects)
        mesh_file = os.path.join(self.scene_dir, f"mesh_{mesh_id}.obj")

        if not os.path.exists(mesh_file):
            print("No Mesh found! Generating new shape: ")
            x, y, z = self._generate_mesh()
            shapes.save_obj_not_overlap(mesh_file, x, y, z)

        mesh = bpy.ops.import_scene.obj(filepath=mesh_file)
        new_obj = list(set(bpy.context.scene.objects) - old_objs)[0]
        return new_obj

    def texture_mesh_from_file(
        self,
        obj=None,
        material_name="texture",
        texture_file="texture.png",
        texture_params={},
        overwrite=True,
        unwrap=True,
        mesh_size=1,
    ):
        """
        Add texture from an image file to a given object.
        """
        # Add material to active object if none specified
        if obj == None:
            obj = self.context.view_layer.objects.active

        print(f"Adding {material_name} material to ", obj)

        self.set_mode("EDIT")

        texture_file = os.path.join(self.scene_dir, "textures", texture_file)

        # Export new random texture if it doesn't exist
        if not os.path.exists(texture_file) or overwrite:
            os.makedirs(os.path.join(self.scene_dir, "textures"), exist_ok=True)

            print("Generating texture")
            min_dot_diam = texture_params.get("min_diam", 50)
            max_dot_diam = texture_params.get("max_diam", 75)
            n_dots = texture_params.get("n_dots", 40)
            height = texture_params.get("height", 1024)
            width = texture_params.get("width", 1024)
            noise = texture_params.get("noise", 0.0)
            replicas = 1
            if noise > 0:
                replicas = self.n_frames
                texture_func = textures.noisy_dot_texture
            else:
                texture_func = textures.dot_texture

            texture = texture_func(
                height=height,
                width=width,
                min_diameter=min_dot_diam,
                max_diameter=max_dot_diam,
                n_dots=n_dots,
                save_file=texture_file,
                replicas=replicas,
                noise=noise,
            )

        if texture_params.get("noise", -1) > 0:
            texture_file += "_0000.png"

        image = self.data.images.load(filepath=texture_file)
        print(f"Adding {image} texture to mesh")
        if unwrap:
            status = bpy.ops.uv.unwrap()
            print("Unwrapped status:", status)
            bpy.ops.uv.cube_project(
                cube_size=mesh_size, clip_to_bounds=True, scale_to_bounds=True
            )

        # Create new texture slot
        mat = self.data.materials.new(name=material_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodeOut = nodes["Material Output"]

        # Add emission shader node
        nodeEmission = nodes.new(type="ShaderNodeEmission")
        nodeEmission.inputs["Strength"].default_value = 5

        # Add image to texture
        texImage = nodes.new("ShaderNodeTexImage")
        texImage.image = image

        if texture_params.get("noise", -1) > 0:
            print("Sequencing Texture")
            texImage.image.source = "SEQUENCE"
            texImage.image_user.frame_duration = self.n_frames
            texImage.image_user.frame_offset = -1

        # Link everything together
        links = mat.node_tree.links
        linkTexture = links.new(texImage.outputs["Color"], nodeEmission.inputs["Color"])
        linkOut = links.new(nodeEmission.outputs["Emission"], nodeOut.inputs["Surface"])

        # Turn off a bunch of material parameters
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Specular"].default_value = 0
        bsdf.inputs["Specular Tint"].default_value = 0
        bsdf.inputs["Roughness"].default_value = 0
        bsdf.inputs["Sheen Tint"].default_value = 0
        bsdf.inputs["Clearcoat"].default_value = 0
        bsdf.inputs["Subsurface Radius"].default_value = [0, 0, 0]
        bsdf.inputs["IOR"].default_value = 0
        mat.cycles.use_transparent_shadow = False

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

            print("adding new material")

        print("...done texturing")
        return

    def add_background_plane(
        self, texture_params={}, overwrite=True, use_image_texture=False
    ):
        """
        Adds plane to background and adds image texture that can be modified during animation.
        The texture material is added with the name "Background" for later access.
        """

        mesh = bpy.data.meshes.new("Plane")
        obj = bpy.data.objects.new("Plane", mesh)

        bpy.context.collection.objects.link(obj)

        bm = bmesh.new()
        bm.from_object(obj, bpy.context.view_layer.depsgraph)

        size = 40
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

        if use_image_texture:
            texture = {}
            texture["min_diam"] = texture_params.get("min_diam", 5)
            texture["max_diam"] = texture_params.get("max_diam", 10)
            texture["n_dots"] = texture_params.get("n_dots", 15000)
            texture["height"] = texture_params.get("height", 2048)
            texture["width"] = texture_params.get("width", 2048)
            texture["noise"] = texture_params.get("noise", 5.0)

            self.texture_mesh(
                material_name="Background",
                texture_file="background",
                overwrite=overwrite,
                texture_params=texture,
                mesh_size=size,
                unwrap=False,
            )

        else:
            scale = texture_params.get("scale", 25)
            randomness = texture_params.get("randomness", 1)
            distance = texture_params.get("distance", "Euclidean")
            colors = texture_params.get("colors", [(0, 0, 0, 1), (1, 1, 1, 1)])
            width = texture_params.get("width", 0.5)

            voronoi_texture(
                scale,
                randomness,
                distance,
                colors,
                width,
                obj,
                material_name="BACKGROUND_PLANE",
            )
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

    def generate_rotation(self, obj, mesh_id=0, save=True, use_existing=True):
        data = {}
        n_frames = self.n_frames
        if os.path.exists(self.rotation_data):
            data = np.load(self.rotation_data, allow_pickle=True).item()

        if (
            not data.get(mesh_id)
            or not use_existing
            or len(data[mesh_id]["quaternion"]) < n_frames
        ):
            print("Generating new rotation")
            data[mesh_id] = {}

            data[mesh_id]["quaternion"] = np.zeros(shape=[n_frames, 4])
            data[mesh_id]["rotation"] = np.zeros(shape=[n_frames, 3, 3])
            data[mesh_id]["angle"] = np.zeros(shape=[n_frames, 1])
            data[mesh_id]["location"] = np.zeros(shape=(n_frames, 3))
            data[mesh_id]["axis"] = np.zeros(shape=[n_frames, 3])

            rotation_axis = np.random.uniform(-1, 1, 3)
            degrees = np.linspace(0, 360, n_frames)

            for frame, degree in enumerate(degrees):
                q = Quaternion(axis=rotation_axis, degrees=degree)

                # Record params
                data[mesh_id]["quaternion"][frame] = q.elements
                data[mesh_id]["rotation"][frame] = q.rotation_matrix
                data[mesh_id]["angle"][frame] = degree
                data[mesh_id]["axis"][frame] = rotation_axis

            if save:
                np.save(self.rotation_data, data, allow_pickle=True)

        scene = self.scene
        scene.frame_start = 1
        scene.frame_end = self.n_frames

        if "quaternion" in data[mesh_id].keys():
            rotations = data[mesh_id]["quaternion"]
        else:
            rotations = data[mesh_id]["rotation"]

        print("=" * 80 + "\n" + str(obj) + "\n" + "=" * 80 + "\n")
        # Set rotation parameters
        obj.rotation_mode = "QUATERNION"

        # Add frame for rotation
        for frame, q in enumerate(rotations):
            obj.rotation_quaternion = q
            obj.keyframe_insert("rotation_quaternion", frame=frame + 1)
            data[mesh_id]["location"] = obj.location

        return data

    def generate_trajectory(self, obj, mesh_id=0, save=True):
        data = {}
        if os.path.exists(self.rotation_data):
            data = np.load(self.rotation_data, allow_pickle=True).item()

        if not data.get(mesh_id):
            data[mesh_id] = {}

        n_frames = self.n_frames
        data[mesh_id]["trajectory"] = np.zeros((n_frames, 3))

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
            data[mesh_id]["trajectory"][i] = obj.location

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
        self.delete_all(obj_type="LIGHT")

        # Set direct and ambient light
        camera_loc = bpy.data.objects["Camera"].location
        camera_rot = bpy.data.objects["Camera"].rotation_euler
        self.set_light_source("SUN", camera_loc, camera_rot)

        if args.background_style == "white":
            world_nodes = self.data.worlds["World"].node_tree.nodes
            world_nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
            world_nodes["Background"].inputs["Strength"].default_value = 1.5
            self.set_background_color(color=(255, 255, 255, 1))

        if args.scene_type == "galaxy":
            bpy.ops.object.empty_add(
                type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
            )
            center_axis = bpy.context.active_object
            self.generate_rotation(center_axis, mesh_id=-1)

        for mesh_id in range(self.n_shapes):
            obj = self.load_mesh(mesh_id=mesh_id)
            if args.scene_type == "galaxy":
                obj.location = np.random.randint(-5, 5, 3)
                obj.parent = center_axis

            texture_file = "texture.png" if args.texture_noise <= 0 else "texture"
            if not args.no_texture_mesh:
                self.texture_mesh(
                    obj=obj,
                    material_name=texture_file,
                    texture_file=texture_file,
                    overwrite=True,
                    texture_params={"noise": args.texture_noise},
                )

            self.generate_rotation(obj, mesh_id=mesh_id, use_existing=True)

            if args.trajectory:
                trajectory_data = self.generate_trajectory(obj, mesh_id=mesh_id)
                self.generate_trajectory(
                    obj, trajectory_data[mesh_id]["trajectory"], mesh_id=mesh_id
                )

        """
        for obj in self.objects:
            obj.select_set(True)
        bpy.ops.view3d.camera_to_view_selected()
        """

        if args.background_style == "textured":
            self.add_background_plane(
                texture_params={"noise": args.background_noise},
                overwrite=args.new_background,
            )

        for obj in self.objects:
            obj.select_set(True)

        bpy.ops.wm.save_as_mainfile(
            filepath=self.scene_dir + "/scene.blend", check_existing=False
        )
        self.render()

        # Clean up scene
        self.delete_all(obj_type="MESH")


def main(args):
    if not os.path.exists(args.root_dir):
        os.mkdir(args.root_dir)
        print("Created root directory: ", args.root_dir)

    with open(args.global_scene_params, "r") as f:
        global_scene_params = json.load(f)
        global_scene_params = global_scene_params[args.experiment_name]

    for scene_num in range(args.start_scene, args.start_scene + args.n_scenes):
        scene_dir = os.path.join(args.root_dir, "scene_%03d" % scene_num)
        logging.info("Processing scene: {}...".format(scene_dir))

        # Create a scene and initialize some basic properties
        scene = BlenderScene(
            scene_dir,
            render_size=args.render_size,
            device=args.device,
            n_frames=args.n_frames,
            engine=args.engine,
            samples=args.samples,
        )
        scene.n_shapes = args.n_shapes
        scene.n_frames = args.n_frames

        # Set camera properties for scene
        camera = scene.data.objects["Camera"]
        camera.location = global_scene_params["camera_location"]
        camera.rotation_euler = global_scene_params["camera_rotation_euler"]
        camera.data.sensor_width = global_scene_params["camera_sensor_width"]
        scene.render_frame_boundaries = copy.deepcopy(
            global_scene_params["render_frame_boundaries"]
        )
        scene.x_len = (
            scene.render_frame_boundaries["top_right"][0]
            - scene.render_frame_boundaries["top_left"][0]
        )
        scene.y_len = (
            scene.render_frame_boundaries["top_right"][1]
            - scene.render_frame_boundaries["bottom_right"][1]
        )

        scene.create_default_scene(args)

        # Add a render timestamp
        ts = time.time()
        fts = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(scene_dir, "timestamp.txt"), "w") as f:
            f.write("Files Rendered at: {}\n".format(fts))


if __name__ == "__main__":
    parser = BlenderArgparse.ArgParser()
    parser.add_argument(
        "--root_dir", type=str, help="Output directory for data", default="scenes"
    )
    parser.add_argument(
        "--global_scene_params",
        type=str,
        help="Path to global scene parameters json file",
        default="/home/yyf/CommonFate/global_scene_params.json",
    )
    parser.add_argument(
        "--n_scenes", type=int, help="Number of scenes to generate", default=867
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

    # Scene settings
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="whether or not to generate trajectory for shapes",
    )
    parser.add_argument(
        "--scene_type",
        type=str,
        default="default",
        help="Type of scene (current options: default | galaxy)",
    )
    parser.add_argument(
        "--n_shapes", type=int, default=1, help="How many meshes to include in scene"
    )

    # Texture default settings:
    min_dot_diam = 100
    max_dot_diam = 200
    n_dots = np.random.randint(10, 20)
    parser.add_argument(
        "--no_texture_mesh",
        action="store_true",
        help="Whether or not to add dot texture to meshes",
    )
    parser.add_argument(
        "--min_dot_diam",
        type=int,
        help="minimum diamater for dots on texture image",
        default=min_dot_diam,
    )
    parser.add_argument(
        "--max_dot_diam",
        type=int,
        help="maximum diamater for dots on texture image",
        default=max_dot_diam,
    )
    parser.add_argument(
        "--n_dots", type=int, help="Number of dots on texture image", default=n_dots
    )
    parser.add_argument(
        "--texture_noise",
        type=float,
        help="Generates noisy texture sequence",
        default=0.0,
    )

    # Render settings
    parser.add_argument(
        "--device", type=str, help='Either "CUDA" or "CPU"', default="CUDA"
    )
    parser.add_argument("--engine", type=str, help="rendering engine", default="CYCLES")
    parser.add_argument(
        "--output_img_name", type=str, help="Name for output images", default="img"
    )
    parser.add_argument(
        "--render_size", type=int, help="size of .png file to render", default=512
    )
    parser.add_argument(
        "--texture_only",
        action="store_true",
        help="Will output only textured mesh, and no rendering",
    )
    parser.add_argument(
        "--samples", type=int, default=256, help="Samples to use in rendering"
    )

    # Background texture settings
    parser.add_argument(
        "--background_style",
        type=str,
        default="textured",
        help="Options: white | textured | none",
    )
    parser.add_argument(
        "--background_noise",
        type=float,
        help="amount of noise in textured background plane",
        default=0.0,
    )
    parser.add_argument(
        "--new_background", action="store_true", help="Generate new background sequence"
    )
    args = parser.parse_args()

    main(args)
