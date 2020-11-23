import bpy
import math

scene = bpy.data.scenes["Scene"]
mycube = bpy.data.objects['mesh.160']
mycube.rotation_mode = 'XYZ'

scene.frame_start = 1
scene.frame_end = 100

mycube.rotation_euler = (0, 0, 0)
mycube.keyframe_insert('rotation_euler', index=2 ,frame=1)

mycube.rotation_euler = (0, 0, math.radians(180))
mycube.keyframe_insert('rotation_euler', index=2 ,frame=100)

scene.render.use_stamp = 0
scene.render.stamp_background = (0,0,0,1)

scene.render.filepath = "/Users/yoni/Desktop/rendered"
scene.render.image_settings.file_format = "AVI_JPEG"
bpy.ops.render.render(animation=True)