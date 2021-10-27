import os
import bpy
from scenes import BlenderScene


# Render all the relevant views

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

    scene.create_default_scene(args)