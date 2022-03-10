from tqdm import tqdm
import argparse
import numpy as np
import os
from glob import glob
import cv2
import array
import sys
from pathlib import Path
from PIL import Image



p = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
p.add_argument('--trgt_path', type=str, required=True, help='The path the output will be dumped to.')
p.add_argument('--extrinsics_dir', type=str, default="/home/yyf/CommonFate", help='The path the output will be dumped to.')
p.add_argument("--images_as_RGB", action="store_true", help="whether or not to read images in RGB (Default is BGR)")
p.add_argument("--reverse_normal_sphere", action="store_true", help="reverses coloring to a more blue color")
argv = sys.argv
opt = p.parse_args()


def exr2numpy(exr, maxvalue=15., normalize=False):
    """ converts 1-channel exr-data to 2D numpy arrays
        Params:
            exr: exr file path
            maxvalue: max clipping value
            RGB: whether to return images in RGB mode (default is BGR)
            normalize: whehter or not to normalize images
    """
    # normalize
    data = cv2.imread(exr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if opt.images_as_RGB:
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = np.array(data)
    data[data > maxvalue] = maxvalue
    data[data == maxvalue] = 0.

    if normalize:
        data /= np.max(data)

    # img = np.transpose(np.array(data)).reshape(data.shape[0], data.shape[1], data.shape[2])
    return data


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()

def main(trgt_path):
    scenes = os.listdir(trgt_path)
    scenes = sorted(scenes)
    for scene in scenes:
        print("Processing scene: ", scene)
        normal_dir = os.path.join(trgt_path, scene, "normals")
        existing_pngs = Path(normal_dir).glob("*.png")
        all_normal_imgs = Path(normal_dir).glob("*.exr")


        for normal_img in tqdm(all_normal_imgs):
            fname = normal_img.stem

            trgt_dir = Path(normal_dir)
            trgt_dir.mkdir(parents=False, exist_ok=True)

            out_fname = trgt_dir / fname

            np_array = exr2numpy(str(normal_img))
            extrinsics_fname =  str(opt.extrinsics_dir) + "/camera_extrinsics.txt"

            extrinsics = load_pose(extrinsics_fname)
            if opt.reverse_normal_sphere:
                extrinsics = extrinsics * -1

            rotation = np.linalg.inv(extrinsics[:3, :3])
            rotated = np.einsum('ij,abj->abi', rotation, np_array)
            coloring = ((rotated * 0.5 + 0.5) * 255)

            rgba = np.concatenate((coloring, np.ones_like(coloring[:, :, :1]) * 255), axis=-1)
            rgba[np.logical_and(rgba[:, :, 0] == 127.5, rgba[:, :, 1] == 127.5, rgba[:, :, 2] == 127.5), :] = 0.

            rotatedImg = Image.fromarray(np.uint8(rgba))

            rotatedImg.save(str(out_fname) + ".png")

if __name__ == '__main__':
    main(opt.trgt_path)
