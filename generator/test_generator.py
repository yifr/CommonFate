import sys
import pathlib

generator_path = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(generator_path)

from render_scenes import BlenderScene

s = BlenderScene('test')