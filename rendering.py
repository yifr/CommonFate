import os
import bpy


class RenderEngine:
    """
    Defines the renderer used for creating videos.
    By default it's set to Blender's Cycles engine,
    which is able to run headless and makue use of GPUs
    for high quality, parallel, rapid rendering on a
    computer cluster
    """

    def __init__(
        self, scene, device="CUDA", engine="CYCLES", render_size=(256), samples=256
    ):
        """
        Params:
            scene: (BlenderScene): Scene which the rendering engine powers
            device: (str): whether to render on CUDA or on a CPU
            engine (str): Which Blender renderer to use
            render_size (int): Output size of rendering engine
            samples (int): Number of samples to take for rendering -- higher == better
            output_dir ()
        """
        self.scene = scene
        self.device = device
        self.engine = engine
        self.render_size = render_size
        self.samples = samples

        if self.device == "CUDA":
            self.activated_gpus = self.enable_gpus(scene)
            print(f"Using following GPUs: {self.activated_gpus}")

    def render(self, output_dir):
        """
        Renders the video to a given folder.
        """
        if not output_dir.endswith("Image"):
            output_dir = os.path.join(output_dir, "Image")

        self.set_render_settings()
        self.scene.render.filepath = output_dir
        bpy.ops.render.render(animation=True)

    def set_render_settings(self):
        # Set properties to increase speed of render time
        scene = self.scene
        scene.render.engine = self.engine  # use cycles for headless rendering
        if len(self.render_size) > 1:
            scene.render.resolution_x = self.render_size[0]
            scene.render.resolution_y = self.render_size[1]
        else:
            scene.render.resolution_x = self.render_size[0]
            scene.render.resolution_y = self.render_size[0]

        scene.render.image_settings.color_mode = "BW"
        scene.render.image_settings.compression = 0
        scene.cycles.samples = self.samples

    def enable_gpus(self, scene, device_type="CUDA", use_cpus=False):
        """
        Sets device as GPU and adjusts rendering tile size accordingly
        """
        scene.render.engine = self.engine  # use cycles for headless rendering
        scene.cycles.device = "GPU"

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cycles_preferences.compute_device_type = device_type

        activated_gpus = []
        print(cycles_preferences.get_devices())
        for device in cycles_preferences.devices:
            print("Activating: ", device)
            device.use = True
            activated_gpus.append(device.name)

        cycles_preferences.compute_device_type = device_type

        scene.render.tile_x = 128
        scene.render.tile_y = 128
        return activated_gpus
