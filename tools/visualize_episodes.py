# example: python tools/visualize_episodes.py --dataset-path /path/to/r2d2.hdf5 --prompt-trajectory-name real_episode_2024-06-01-poke-tiger_1
import h5py
import tyro 
import numpy as np
import imgviz
import os

def main(
    dataset_path : str, # ends with r2d2.hdf5,
    prompt_trajectory_name : str # i.e. real_episode_2024-06-01-poke-tiger_1
):
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."
    dataset = h5py.File(dataset_path, 'r')
    assert prompt_trajectory_name in dataset.keys(), f"Trajectory key \"{prompt_trajectory_name}\" not found in dataset."

    # get the goal observation of the dataset 
    side_images = dataset[f"{prompt_trajectory_name}/observation/exterior_image_1_left"][:]
    wrist_images = dataset[f"{prompt_trajectory_name}/observation/wrist_image_left"][:]

    side_images_l, wrist_images_l = [], []
    for si, wi in zip(side_images, wrist_images):
        if len(si) == 0 or len(wi) == 0:
            continue
        side_images_l.append(np.frombuffer(si, dtype="uint8").reshape(180, 320, 3))
        wrist_images_l.append(np.frombuffer(wi, dtype="uint8").reshape(180, 320, 3))

    for i, (side_img, wrist_img) in enumerate(zip(side_images_l, wrist_images_l)):
        im_tile = imgviz.tile(
            [side_img, wrist_img],
            shape=(1, 2),
            border=(255, 255, 255),
        )
        imgviz.io.cv_imshow(im_tile)
        key = imgviz.io.cv_waitkey(1000 // 30)
        if key == ord('q'):
            break

if __name__ == "__main__":
    tyro.extras.set_accent_color("yellow")
    tyro.cli(main)