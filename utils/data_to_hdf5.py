import os
import sys
from PIL import Image
import numpy as np
import h5py as hp
import argparse
import tqdm
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, help="Root dataset directory")
parser.add_argument("--dataset_name", type=str, help="Name of the output hdf5 file")

args = parser.parse_args()


def clevr_to_hdf5():
    dataset_root = os.path.join(args.root_dir, "CLEVR_v1.0", "images")

    fname = os.path.join(args.root_dir, args.dataset_name + ".hdf5")
    f = hp.File(fname, "w")

    group = f.create_group("images")
    data_splits = ["train", "test", "val"]

    for data_split in data_splits:
        data_path = os.path.join(dataset_root, data_split)
        image_names = os.listdir(data_path)
        print(f"{len(image_names)} images in {data_split} split")
        sys.stdout.flush()

        dataset_shape = (len(image_names), 3, 256, 256)
        dataset = group.create_dataset(data_split, dataset_shape)

        for i, img_name in tqdm.tqdm(enumerate(image_names)):
            img_path = os.path.join(data_path, img_name)
            img = np.asarray(Image.open(img_path).convert("RGB").resize((256, 256)))
            img = np.transpose(img, [2, 0, 1]) / 255
            dataset[i] = img

        # print(f"{data_split} dataset statistics: ")
        # print("Channel Mean: ", dataset.mean(axis=(0, 2, 3)))
        # print("Channel Std: ", dataset.std(axis=(0, 2, 3)))

    f.close()


def gestalt_to_hdf5():
    """
    Generates hdf5 file per scene for DorsalVentral model (based on TDW dataloader)
    """
    dataset_root = os.path.join(args.root_dir, args.dataset_name)
    os.makedirs(os.path.join(args.root_dir, args.dataset_name + "_hdf5"), exist_ok=True)
    hdf5_root = os.path.join(args.root_dir, args.dataset_name + "_hdf5")
    print("Created directory: ", hdf5_root)

    for scene_name in os.listdir(dataset_root):
        if not scene_name.startswith("scene"):
            continue

        fname = os.path.join(hdf5_root, scene_name + ".hdf5")
        f = hp.File(fname, "w")
        frames = f.create_group("frames")

        scene_images = os.path.join(dataset_root, scene_name, "images")
        image_names = os.listdir(scene_images)
        gestalt_images = [image for image in image_names if image.startswith("img_")]
        print(f"{len(gestalt_images)} images in {scene_name}")
        print(f"{len(image_names)} total images in {scene_images}")
        sys.stdout.flush()

        img_dataset_shape = (512, 512, 3)
        for i, img_name in tqdm.tqdm(enumerate(gestalt_images)):
            frame_group = frames.create_group(f"{i:04d}")
            image_group = frame_group.create_group("images")
            dataset = image_group.create_dataset(
                "_img", img_dataset_shape, dtype=np.uint8
            )
            img_path = os.path.join(scene_images, img_name)
            img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)

            dataset[:, :, :] = img[:, :, :]

        f.close()

    print("Done")


# generate gestalt hdf5 files:
# for each scene, generate a hdf5 file with the following structure:
# add_group("scene_num")
# initialize np array (shape: (n_frames, 3, size, size))
# for each image, add np.array to dataset

if __name__ == "__main__":
    gestalt_to_hdf5()
