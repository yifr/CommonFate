import numpy as np
from PIL import Image, ImageDraw


def dot_texture(
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


def ordered_texture(
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


def noisy_dot_texture(
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


if __name__ == "__main__":
    ordered_texture()
