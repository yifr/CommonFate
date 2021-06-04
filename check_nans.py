from data_loader import SceneLoader
from models.cnn import ShapeNet

s = SceneLoader(root_dir="scenes/single_shape_textured_v2", n_frames=20, device="cpu")
model = ShapeNet(out_size=2)

for i in range(1000):
    data = s.get_scene(i)
    shape_params = data["shape_params"].mean(axis=0)
    inv = model.inverse_transform(shape_params)
    print(inv)
