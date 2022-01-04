from tqdm import tqdm
import argparse
import numpy as np
import os
from glob import glob
import OpenEXR
import Imath
import array
import sys
from pathlib import Path
from PIL import Image



p = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
p.add_argument('--trgt_path', type=str, required=True, help='The path the output will be dumped to.')
argv = sys.argv
opt = p.parse_args()


def exr2numpy(exr, maxvalue=15.,normalize=False):
    """ converts 1-channel exr-data to 2D numpy arrays """
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (RGB) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("RGB") ]

    # create numpy 2D-array
    img = np.zeros((sz[1],sz[0],3), np.float64)

    # normalize
    data = np.array(RGB)
    data[data > maxvalue] = maxvalue
    data[data == maxvalue] = 0.

    if normalize:
        data /= np.max(data)

    img = np.transpose(np.array(data)).reshape(img.shape[0],img.shape[1],img.shape[2])

    return img


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
        
        outDir = base_dir / 'sorryThoseNormalsDontLookNiceButTheyMightJustWork'
        out_fname = outDir / fname
        
        outDir.mkdir(parents=False, exist_ok=True)

        np_array = exr2numpy(str(depth_img))
       
        #im = Image.fromarray(np.uint8(np_array*255))
        #im.save(str(trgt_fname)+".png")
        
        extrinsics_fname =  str(base_dir / 'pose' / fname)+".txt"

        extrinsics = load_pose(extrinsics_fname)
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
