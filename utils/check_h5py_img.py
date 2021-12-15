import h5py as hp
from PIL import Image

fname = "scenes/gestalt_v3_hdf5/scene_001.hdf5"
f = hp.File(fname, "r")
data = f["frames"]["0001"]["images"]["_img"][:].transpose(2, 1, 0)
print(data.shape)

image = Image.fromarray(data, "RGB")
image.save("test2.png")
f.close()
