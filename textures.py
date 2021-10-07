import numpy as np
from PIL import Image, ImageDraw

try:
    import bpy
except ImportError:
    print(
        "Unable to import Blender Python Interface (bpy). \
        No procedural textures will be available."
    )

TEXTURE_FUNCTIONS = [
    "base_texture",
    "transparent_texture",
    "glass_texture",
    "dot_texture_png",
    "ordered_texture_png",
    "noisy_dot_texture_png",
]


PROCEDURAL_TEXTURES = ["voronoi", "wave", "magic", "checker", "noise", "brick"]


def Shader(name):
    return "ShaderNodeTex" + name.capitalize()


TEXTURE_MAPS = {
    "ShaderNodeTexVoronoi": {"Scale": [1.25, 10], "Randomness": [0, 1]},
    "ShaderNodeTexWave": {
        "Scale": [0.25, 3],
        "Distortion": [0, 7.5],
        "Detail": [0, 5],
        "Detail Scale": [0, 5],
    },
    "ShaderNodeTexNoise": {
        "Scale": [1, 10],
        "Detail": [1, 2],
        "Roughness": [0, 1],
        "Distortion": [0, 4],
    },
    "ShaderNodeTexChecker": {"Scale": [2, 30]},
    "ShaderNodeTexBrick": {
        "Scale": [1, 10],
        "Mortar Size": [0, 0.025],
        "Mortar Smooth": [0, 1],
        "Bias": [-1, 0],
        "Brick Width": [0.02, 2],
        "Row Height": [0.25, 1],
    },
    "ShaderNodeTexMagic": {"Scale": [2, 5], "Distortion": [0.5, 10]},
}


def base_texture(
    scene,
    texture_type,
    texture_params,
    material_color=None,
    width=0.5,
    obj=None,
    material_name="texture",
):
    """
    Appends procedural texture polka dot texture to mesh.
    Params:
        scene: (BlenderScene): The BlenderScene in which the mesh lives
        scale: (int) controls the number of dots
        randomness: (int) controls how random the dots are arranged
        distance: (str) controls distance function for texture texture
        colors: (tuple list) colors of texture
        width: relative size of dots
        obj:
    """
    if obj == None:
        obj = scene.context.view_layer.objects.active

    print(f"Adding {material_name} material to ", obj)

    scene.set_mode("EDIT")

    mat = scene.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]

    bsdf.inputs["Specular"].default_value = 0
    bsdf.inputs["Roughness"].default_value = 1
    bsdf.inputs["Transmission"].default_value = 0
    bsdf.inputs["Sheen Tint"].default_value = 0
    mat.shadow_method = "NONE"

    texture = nodes.new(type=texture_type)
    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    mapping_node = nodes.new(type="ShaderNodeMapping")
    coordinate_node = nodes.new(type="ShaderNodeTexCoord")

    ##############
    # Link Nodes
    ##############
    links = mat.node_tree.links

    # Coordinate Texture -> Mapping Node --> Noise Texture
    links.new(coordinate_node.outputs["Object"], mapping_node.inputs[0])
    links.new(mapping_node.outputs[0], texture.inputs[0])

    links.new(texture.outputs[0], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    color_ramp.color_ramp.elements.new(0.5)

    color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)

    # Evenly interpolate between color/white spots
    color_ramp.color_ramp.elements[0].position = 0
    color_ramp.color_ramp.elements[1].position = width

    color_ramp.color_ramp.interpolation = "CONSTANT"

    if material_color:
        bsdf.inputs["Base Color"].default_value = material_color

    for param in texture_params:
        param_val = texture_params[param]
        if type(param_val) == list:
            param_val = np.random.uniform(param_val[0], param_val[1])
        try:
            texture.inputs[param].default_value = param_val
        except KeyError:
            print(
                f"Invalid texture parameter: {param} for texture type: {texture_type}."
            )

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    print(f"Added {texture_type} Texture...")
    return


def transparent_texture(
    scene,
    texture_type="ShaderNodeTexMagic",
    texture_params={},
    material_name="transparent_bands",
    material_color=None,
    obj=None,
):
    """
    Creates transparent holes in objects. The shape of the transparent regions
    can be controlled by modifying the parameters of the `texture_type`.

    Params:
        scene: (BlenderScene):
                            Current Blender Scene
        noise_texture: (str):
                            Controls shape of transparent regions.
                            Must be the correct name of a ShaderNode
                            (ie; 'ShaderNodeTextexture' | 'ShaderNodeTexMagic')
        texture_params: (dict):
                            Parameters to change on noise texture.
        material_name: (str):
                            Name for material
        material_color: (tuple->ints):
                            Optional color for base material. Must
                            Defaults to white.
        obj: (bpy object):
                            Pointer to blender object. Defaults to active object
    """
    if obj == None:
        obj = scene.context.view_layer.objects.active

    # scene.set_mode("EDIT")

    mat = scene.data.materials.new(name=material_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]

    ###########################################################
    # Set some default BSDF settings
    # to make sure material isn't too transparent or glossy
    ##########################################################
    bsdf.inputs["Specular"].default_value = 0
    bsdf.inputs["Roughness"].default_value = 1
    bsdf.inputs["Transmission"].default_value = 0
    bsdf.inputs["Sheen Tint"].default_value = 0
    mat.shadow_method = "NONE"

    ##############
    # Create Nodes
    ##############
    mix_shader = nodes.new(type="ShaderNodeMixShader")
    transparent_bsdf = nodes.new(type="ShaderNodeBsdfTransparent")
    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    try:
        noise_texture = nodes.new(type=texture_type)
    except ValueError:
        print(
            f"Texture Type: '{texture_type}' is not a valid ShaderNode. \
            Please check the BPY manual (https://docs.blender.org/api/current/bpy.types.ShaderNode.html) \
            to find a valid texture type. Defaulting to MagicTexture"
        )
        noise_texture = nodes.new(type="ShaderNodeTexMagic")

    mapping_node = nodes.new(type="ShaderNodeMapping")
    coordinate_node = nodes.new(type="ShaderNodeTexCoord")

    ##############
    # Link Nodes
    ##############
    links = mat.node_tree.links

    # Coordinate Texture -> Mapping Node --> Noise Texture
    links.new(coordinate_node.outputs["Object"], mapping_node.inputs[0])
    links.new(mapping_node.outputs[0], noise_texture.inputs[0])

    # Noise texture -> Color Ramp
    links.new(noise_texture.outputs[0], color_ramp.inputs["Fac"])

    # Connect Everything to Mix Shader
    links.new(color_ramp.outputs["Color"], mix_shader.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])
    links.new(transparent_bsdf.outputs["BSDF"], mix_shader.inputs[2])

    # Mix Shader --> Output
    links.new(mix_shader.outputs[0], nodes["Material Output"].inputs["Surface"])

    ########################
    # Modify texture params
    ########################

    # Change base color
    if material_color:
        bsdf.inputs["Base Color"].default_value = material_color

    # Set color ramp to constant black/white with even interpolation
    color_ramp.color_ramp.interpolation = "CONSTANT"
    color_ramp.color_ramp.elements.new(0.5)
    color_ramp.color_ramp.elements[0].position = 0
    color_ramp.color_ramp.elements[1].position = 0.5
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)

    transparent_bsdf.inputs["Color"].default_value = (1, 1, 1, 1)

    # Modify noise texture params
    for param in texture_params:
        try:
            noise_texture.inputs[param].default_value = texture_params[param]
        except KeyError:
            print(
                f"Invalid texture parameter: {param} for texture type: {texture_type}."
            )

    if scene.renderer.engine == "BLENDER_EEVEE":
        mat.blend_method = "HASHED"

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    print(f"Added Transparent Texture to {obj}...")
    return


def remove_bsdf_props(self, mat):
    """
    Removes unneeded properties from Principled BSDF
    Params:
        mat: (Blender Material):
            Material to remove properties from
    """
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
    return


def glass_texture(self):
    if self.renderer.engine == "CYCLES":
        # do the stuff that is required to
        # create glass in cycles
        pass
    elif self.render.engine == "EEVEE":
        # do the stuff that is required to
        # create glass in eevee
        pass


def dot_texture_png(
    width=1024,
    height=1024,
    min_diameter=10,
    max_diameter=20,
    n_dots=900,
    replicas=1,
    noise=-1,
    save_file="../scenes/scene_999/textures/texture.png",
):
    """
    Creates white image with a random number of black dots

    Params
    --------
        width: int: width of image
        height: int: height of image
        min_diameter: int: min diameter of dots
        max_diameter: int: max diameter of dots
        n_dot: int: number of dots to create
        save_file: str: if provided, will save the image to given path
    """
    for i in range(replicas):
        img = Image.new("RGB", (width, height), color=(255, 255, 255, 1))
        draw = ImageDraw.Draw(img)
        for _ in range(n_dots):
            x, y = np.random.randint(
                max_diameter, width - max_diameter
            ), np.random.randint(max_diameter, height - max_diameter)
            diam = np.random.randint(min_diameter, max_diameter)
            draw.ellipse([x, y, x + diam, y + diam], fill="black")

        if save_file is not None:
            if noise > 0:
                img.save(save_file + f"_{i:04d}.png", format="png")
            else:
                img.save(save_file, format="png")

    return img


def ordered_texture_png(
    width=1024,
    height=1024,
    min_diameter=30,
    max_diameter=50,
    n_dots=20,
    save_file="test.png",
    **kwargs,
):

    img = Image.new("RGB", (width, height), color=(255, 255, 255, 1))
    draw = ImageDraw.Draw(img)
    xs = np.linspace(max_diameter, width + max_diameter, n_dots)
    ys = np.linspace(max_diameter, width + max_diameter, n_dots)
    for x in xs:
        for y in ys:
            diam = np.random.randint(min_diameter, max_diameter)
            draw.ellipse([x, y, x + diam, y + diam], fill="black")

    if save_file is not None:
        img.save(save_file, format="png")

    return img


def noisy_dot_texture_png(
    width=2048,
    height=2048,
    min_diameter=4,
    max_diameter=8,
    n_dots=15500,
    replicas=360,
    noise=1.0,
    save_file="../scenes/scene_999/textures/background",
):
    """
    Creates white image with a random number of black dots

    Params
    --------
        width: int: width of image
        height: int: height of image
        min_diameter: int: min diameter of dots
        max_diameter: int: max diameter of dots
        n_dots: int: number of dots to create
        replicas: int: how many images to create
        noise: float: range that x, y coordinates vary frame to frame
        save_file: str: if provided, will save the image to given path
    """
    x, y = np.random.randint(
        max_diameter, width - max_diameter, size=n_dots
    ), np.random.randint(max_diameter, height - max_diameter, size=n_dots)
    diam = np.random.randint(min_diameter, max_diameter, size=(n_dots, 2))

    trajectories = np.random.randint(-noise, noise, size=(n_dots, 2))
    for frame in range(replicas):
        print("Generating background ", frame)
        img = Image.new("RGB", (width, height), color=(255, 255, 255, 1))
        draw = ImageDraw.Draw(img)

        trajectory = trajectories * np.random.random(size=(n_dots, 2))
        x, y = x + trajectory[:, 0], y + trajectory[:, 1]

        for i in range(n_dots):
            draw.ellipse(
                [x[i], y[i], x[i] + diam[i, 0], y[i] + diam[i, 1]], fill="black"
            )

        if save_file is not None:
            if noise > 0:
                img.save(save_file + f"_{frame:04d}.png", format="png")
            else:
                img.save(save_file, format="png")

    return img


def add_texture(scene, obj, tex_config):
    texture_config = tex_config.copy()
    texture_type = texture_config.get("type")
    if texture_type == "random":
        texture_type = np.random.choice(PROCEDURAL_TEXTURES)

    if texture_type not in PROCEDURAL_TEXTURES:
        raise ValueError(
            f"""Texture type: {texture_type} is not an option.
            Please specify a texture type from the following list:
            {', '.join(PROCEDURAL_TEXTURES)}"""
        )

    texture_params = texture_config.get("params", {})
    shader = Shader(texture_type)
    shader_params = TEXTURE_MAPS[shader]
    for param in shader_params:
        if param not in texture_params:
            shader_range = shader_params[param]
            texture_params[param] = np.random.uniform(shader_range[0], shader_range[1])

    width = texture_params.get("Width", 0.5)
    material_color = texture_params.get("material_color")
    material_name = texture_params.get("material_name", "texture")

    transparent = texture_config.get("transparent")
    if transparent:
        if not material_color:
            material_color = (0, 0, 0, 1)
        print("Generating Transparent Object texture")
        transparent_texture(
            scene,
            texture_type=shader,
            texture_params=texture_params,
            material_color=material_color,
            obj=obj,
        )
    else:
        base_texture(
            scene,
            texture_type=shader,
            texture_params=texture_params,
            width=width,
            obj=obj,
            material_color=material_color,
            material_name=material_name,
        )

    texture_config["params"] = texture_params
    return texture_config


if __name__ == "__main__":
    ordered_texture_png()
