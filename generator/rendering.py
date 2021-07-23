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
        self, scene, device="CUDA", engine="CYCLES", render_size=256, samples=256
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
        if output_dir[:-4] != "img_":
            output_dir = os.path.join(output_dir, "img_")

        self.set_render_settings()
        self.scene.render.filepath = output_dir
        bpy.ops.render.render(animation=True)

    def set_render_settings(self):
        # Set properties to increase speed of render time
        scene = self.scene
        scene.render.engine = self.engine  # use cycles for headless rendering
        scene.render.resolution_x = self.render_size
        scene.render.resolution_y = self.render_size
        scene.render.image_settings.color_mode = "BW"
        scene.render.image_settings.compression = 0
        scene.cycles.samples = self.samples

    def enable_gpus(self, scene, device_type="CUDA", use_cpus=False):
        """
        Sets device as GPU and adjusts rendering tile size accordingly
        """
        scene.render.engine = self.engine  # use cycles for headless rendering

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons["cycles"].preferences
        cuda_devices, opencl_devices = cycles_preferences.get_devices()

        if device_type == "CUDA":
            devices = cuda_devices
        elif device_type == "OPENCL":
            devices = opencl_devices
        else:
            raise RuntimeError("Unsupported device type")

        activated_gpus = []

        for device in devices:
            if device.type == "CPU":
                device.use = use_cpus
            else:
                device.use = True
                activated_gpus.append(device.name)

        scene.cycles.device = "GPU"
        cycles_preferences.compute_device_type = device_type

        scene.render.tile_x = 128
        scene.render.tile_y = 128

        return activated_gpus
