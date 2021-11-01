import bpy
import bpy_extras
import bpy_types
from mathutils import Vector
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Pixel:
    location: Tuple[int, int]
    visible: bool


def check_coordinate_visibility(
    scene: bpy.types.Scene, source: Vector, target: Vector, threshold=0.0001
) -> bool:
    occluded, location, *_ = scene.ray_cast(
        scene.view_layers[0], origin=source, direction=(target - source)
    )
    self_occluded = (location - target).length < threshold
    visible = occluded is False or self_occluded
    return visible


def get_2d_pixel(coordinate_3d: Tuple[float, float, float], camera_name: str) -> Pixel:
    scene = bpy.context.scene
    camera_obj = bpy.data.objects[camera_name]
    coordinate_3d = Vector(coordinate_3d)
    co_2d = bpy_extras.object_utils.world_to_camera_view(
        scene, camera_obj, coordinate_3d
    )
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_y * render_scale),
        int(scene.render.resolution_x * render_scale),
    )
    pixel_coordinate = (
        round(co_2d.x * render_size[0]),
        round((1 - co_2d.y) * render_size[1]),
    )
    visibility = check_coordinate_visibility(
        scene, source=camera_obj.location, target=coordinate_3d
    )
    return Pixel(pixel_coordinate, visibility)


def get_2d_coordinates_of_object_vertices(
    object_name: str, camera_name: str, only_visibile: bool = True
) -> List[Tuple[int, int]]:
    obj = bpy.data.objects[object_name]
    vertices = [v.co for v in obj.data.vertices]
    pixels = [get_2d_pixel(obj.matrix_world @ v, camera_name) for v in vertices]
    get_all_pixels = only_visibile is False
    coordinates = [
        pixel.location for pixel in pixels if get_all_pixels or pixel.visible
    ]
    return coordinates


def create_keypoints_image(
    keypoints: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    circle_size: int = 3,
):
    import numpy as np
    from skimage.draw import disk

    image = np.ones(image_shape + (3,), dtype=np.uint8) * 255
    for keypoint in keypoints:
        rr, cc = disk(center=keypoint, radius=circle_size, shape=image_shape)
        image[rr, cc] = color
    return image
