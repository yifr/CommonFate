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
argv = sys.argv
opt = p.parse_args()


def exr2numpy(exr, maxvalue=15.,normalize=False):
    """ converts 1-channel exr-data to 2D numpy arrays """
    # normalize
    BGR = cv2.imread(exr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    data = np.array(RGB)
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
    all_depth_imgs = Path(trgt_path).glob('**/*.exr')
    for depth_img in tqdm(all_depth_imgs):
        fname = depth_img.stem

        base_dir = depth_img.parents[0].parents[0]
        trgt_dir = base_dir / 'normals'
        trgt_dir.mkdir(parents=False, exist_ok=True)

        trgt_fname = trgt_dir / fname

        outDir = base_dir / 'np_normals'
        out_fname = outDir / fname

        outDir.mkdir(parents=False, exist_ok=True)

        np_array = exr2numpy(str(depth_img))

        #im = Image.fromarray(np.uint8(np_array*255))
        #im.save(str(trgt_fname)+".png")

        # extrinsics_fname =  str(base_dir / 'pose' / fname)+".txt"
        extrinsics_fname =  str(opt.extrinsics_dir)+"/camera_extrinsics.txt"

        extrinsics = load_pose(extrinsics_fname) * -1
        rotation = np.linalg.inv(extrinsics[:3,:3])
        rotated = np.einsum('ij,abj->abi',rotation,np_array)
        #print("before", rotated[127,127,:])
        #rotated = rotated / (np.linalg.norm(rotated, axis=-1)[:,:,None] + 1e-9)

        #rotated [:,:,0] = -rotated [:,:,0]

        #rotated [:,:,1] = -rotated [:,:,1]
        #print("after",rotated[127,127,:])
        #rotated [:,:,2] = -rotated [:,:,2]


        coloring = ((rotated*0.5+0.5)*255)
        #coloring[:,:,[0,1,2]] = coloring[:,:,[2,1,0]]

        rgba = np.concatenate((coloring, np.ones_like(coloring[:, :, :1])*255), axis=-1)
        #print(rgba)
        rgba[np.logical_and(rgba[:, :, 0] == 127.5, rgba[:, :, 1] == 127.5, rgba[:, :, 2] == 127.5),:] = 0.

        rotatedImg = Image.fromarray(np.uint8(rgba))

        rotatedImg.save(str(out_fname)+".png")

if __name__ == '__main__':
    main(opt.trgt_path)
