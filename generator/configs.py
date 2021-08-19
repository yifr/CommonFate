import numpy as np


def flip(weight=0.5):
    return True if np.random.random() < weight else False


def generate_random_config(max_shapes=6, hierarchy_freq=0.5, transparent_freq=0.2):
    config = {"objects": {}}
    n_objects = np.random.randint(1, max_shapes)
    
    hierarchy = flip(hierarchy_freq)
    if hierarchy:
        transparent = flip(transparent_freq)
        child_params = {
            "shape_type": "random",
            "shape_params": "random",
            "scaling_params": [1, 1, 1, 2],
            "texture": {"type": "random", "transparent": transparent},
            "rotation": flip(0.5),
            "translation": False,
            "location": "random",
        }

        n_points = np.random.randint(low=3, high=5)
        obj_id = "h1"

        object_params = {
            "shape_type": "random",
            "shape_params": "random",
            "scaling_params": [8, 8, 8, 16],
            "n_points": n_points,
            "child_params": child_params,
            "rotation": True,
            "translation": False,
        }

        config["objects"][obj_id] = object_params
        return config

    else:

        for i in range(n_objects):
            obj_id = f"s{i}"
            transparent = flip(transparent_freq)
            rotation = True
            translation = flip()

            object_params = {
                "shape_type": "random",
                "shape_params": "random",
                "scaling_params": [1, 1, 1, 2],
                "texture": {"type": "random", "transparent": transparent},
                "rotation": rotation,
                "translation": False,
                "location": "random",
            }

            config["objects"][obj_id] = object_params

    return config


# To-Dos
# 1) Generate translation
