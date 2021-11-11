import bpy

ops = bpy.ops
data = bpy.data
context = bpy.context

objects = data.objects
scene = context.scene

scene.use_nodes = True
scene.render.engine = "CYCLES"

# Give each object in the scene a unique pass index
scene.view_layers["View Layer"].use_pass_object_index = True
for i, object in enumerate(objects):
    object.pass_index = i

# Set up the nodes
node_tree = scene.node_tree
nodes = node_tree.nodes
render_layer = nodes["Render Layers"]

file_output = nodes.new(type="CompositorNodeOutputFile")
file_output.layer_slots.new("Segmentation")

# Change output path for segmentation images
path = file_output.base_path
path_sections = path.split('/')[:-1]
new_path = '/'.join(path_sections) + '/segmentation_'
file_output.base_path = new_path

# Make sure pass indexes are normalized to [0, 1]
div = nodes.new(type="CompositorNodeMath")
div.operation = "DIVIDE"
div.inputs[1].default_value = 255

# link everything up
node_tree.links.new(render_layer.outputs["Image"], file_output.inputs["Image"])
node_tree.links.new(render_layer.outputs['IndexOB'], div.inputs[0])
node_tree.links.new(div.outputs[0], file_output.inputs["Segmentation"])

scene.render()