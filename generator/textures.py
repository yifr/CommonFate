import numpy as np
from PIL import Image, ImageDraw

def dot_texture(width=1024, height=1024,
                min_diameter=10, max_diameter=20,
                n_dots=900, replicas=1, sequence=False,
                save_file=None):
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

        img  = Image.new('RGB', (width, height), color=(255, 255, 255, 1))
        draw = ImageDraw.Draw(img)
        for _ in range(n_dots):
            x, y = np.random.randint(max_diameter,width - max_diameter - 10), np.random.randint(max_diameter, height - max_diameter - 10) 
            diam = np.random.randint(min_diameter, max_diameter)
            draw.ellipse([x,y,x+diam,y+diam], fill='black')

        if save_file is not None:
            if sequence:
                    img.save(save_file + f'_{i:04d}.png', format='png')
            else:
                img.save(save_file, format='png')
            
    return img
