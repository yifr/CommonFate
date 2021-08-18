import numpy as np


def flip(weight=0.5):
    return True if np.random.random() < weight else False


def generate_random_config(max_shapes=6, hierarchy_freq=0.7, transparent_freq=0.4):
    config = {"objects": {}}
    n_objects = np.random.randint(1, max_shapes)
    n_hierarchies = 0
    for i in range(n_objects):
        obj_id = f"s{i}"
        transparent = flip(transparent_freq)
        rotation = flip(0.9)
        if not rotation:
            translation = True
        else:
            translation = flip()

        object_params = {
            "shape_type": "random",
            "shape_params": "random",
            "scaling_params": [1, 1, 1, 2],
            "texture": {"type": "random", "transparent": transparent},
            "rotation": rotation,
            "translation": translation,
            "location": "random",
        }

        hierarchy = flip(hierarchy_freq)
        if hierarchy:
            n_hierarchies += 1
            n_points = np.random.randint(3, 5)
            obj_id = f"h{n_hierarchies}"
            child_params = object_params.copy()

            rotation = flip(0.9)
            if not rotation:
                translation = True
            else:
                translation = flip()

            object_params = {
                "shape_type": "random",
                "shape_params": "random",
                "scaling_params": [8, 8, 8, 16],
                "n_points": n_points,
                "child_params": child_params,
                "rotation": rotation,
                "translation": translation,
            }

        config["objects"][obj_id] = object_params

    return config


# To-Dos
# 1) Generate translation
