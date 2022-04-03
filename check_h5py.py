import h5py
import os

root = "/om2/user/yyf/CommonFate/scenes"
texs = ["voronoi", "noise"]
splits = ["superquadric_1", "superquadric_2", "superquadric_3", "superquadric_4"]
for tex in texs:
    for split in splits:
        path = os.path.join(root, tex, split + ".hdf5")
        try:
            with h5py.File(path, "r", swmr=True) as f:
                print(path, len(list(f.keys())))
        except:
            print("file busy")
