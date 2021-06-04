from data_loader import SceneLoader
from models.cnn import ShapeNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dirs",
    type=str,
    nargs="+",
    default=["scenes/single_shape_plain", "scenes/single_shape_textured_v2"],
)
args = parser.parse_args()
root_dirs = args.root_dirs
# root_dirs = ['scenes/single_shape_textured_v2', 'scenes/single_shape_plain'],
s = SceneLoader(root_dirs=root_dirs, n_scenes=2000, n_frames=20, device="cpu")
print(len(s.train_idxs) * 20)
model = ShapeNet(out_size=4, conv_dims=3)

for i in range(1):
    data = s.get_scene(i)
    frames = data["frame"].reshape(1, 20, 256, 256).unsqueeze(0)
    out = model(frames)
    # shape_params = data['shape_params'].mean(axis=0)
    # inv = model.inverse_transform(shape_params)
    print(out)
