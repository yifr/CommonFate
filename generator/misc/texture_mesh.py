import bpy


def texture(img):
    """
    Texture by wrapping an image around a mesh
    Assumes only one mesh active at a time
    Params:
        img: :obj: opened blender image
    """
    # Create UV map using a cube projection
    cube_project()

    # Create new texture slot
    mat = bpy.data.materials.new(name="texture")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    # Add image to texture
    texImage = mat.node_tree.nodes.new("ShaderNodeTexImage")
    texImage.image = img
    mat.node_tree.links.new(bsdf.inputs["Base Color"], texImage.outputs["Color"])

    ob = bpy.context.view_layer.objects.active

    # Assign texture to object
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)


def set_mode(mode):
    scene = bpy.context.scene
    # scene.layers = [True] * 20 # Show all layers

    for obj in scene.objects:
        if obj.type == "MESH":
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode=mode)


def cube_project():
    set_mode("EDIT")
    bpy.ops.uv.cube_project(cube_size=1)


def delete():
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.select_set(True)
        else:
            o.select_set(False)

    # Call the operator only once
    set_mode("OBJECT")
    bpy.ops.object.delete()


def load_img(path):
    img = bpy.data.images.load(filepath=path)
    return img


def load_obj(path):
    mesh = bpy.ops.import_scene.obj(filepath=path)
    return mesh


def export_obj(obj, scene_dir):
    """
    Exports textured mesh to a file called textured.obj.
    Also exports material group used to texture
    Assumes one active mesh at a time
    """
    set_mode("OBJECT")
    output_file = scene_dir + "/textured.obj"
    bpy.ops.export_scene.obj(filepath=output_file, use_selection=True)


def main():
    delete()
    set_mode("OBJECT")
    n_scenes = 143
    base_dir = "objects"
    for scene_num in range(n_scenes):
        scene_dir = base_dir + "/scene_%03d" % scene_num
        obj_file = scene_dir + "/mesh.obj"
        texture_file = scene_dir + "/texture.jpg"

        img = load_img(texture_file)
        mesh = load_obj(obj_file)

        texture(img)
        export_obj(mesh, scene_dir)
        delete()


if __name__ == "__main__":
    main()
