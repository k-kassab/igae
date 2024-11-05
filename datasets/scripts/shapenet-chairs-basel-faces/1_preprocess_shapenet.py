# Usage: python datasets/scripts/preprocess_shapenet.py --source /home/a.schnepf/phd/data/shapenet/cars

import json
import numpy as np
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import cv2

def open_cv_to_open_gl(cam_matrix):
        '''
        Transform camera transformation matrix axis.
        '''
        reverse = np.diag([1, -1, -1, 1])
        cam_matrix =  cam_matrix @ reverse
        return cam_matrix


def open_gl_to_open_cv(cam_matrix):
        return open_cv_to_open_gl(cam_matrix)


def list_recursive(folderpath):
    return [os.path.join(folderpath, filename) for filename in os.listdir(folderpath)]

def get_altitude(extrinsics):
    return extrinsics[2, 3]

def cam_inv(camera_matrix):
    """Function to efficiently invert between cam2world and world2cam"""
    assert camera_matrix.shape == (4, 4)
    inverse_camera_matrix = np.eye(4)
    
    R = camera_matrix[:3, :3]
    t = camera_matrix[:3, 3]
    inverse_camera_matrix[:3, :3] = R.T
    inverse_camera_matrix[:3, 3] = -R.T @ t
    return inverse_camera_matrix


if __name__ == '__main__':
    if os.environ['USER'] == "k.kassab":
        root_data = Path("/home/k.kassab/3da-ae-data")
    elif os.environ['USER'] == "a.schnepf":
        root_data = Path("/home/a.schnepf/phd/data")
    else:
        raise ValueError
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_path", type=str, default="shapenet/chairs/raw")
    parser.add_argument("--destination_path", type=str, default="shapenet/chairs/square128_above")
    parser.add_argument("--target_resolution", type=int, default=128)
    parser.add_argument("--og_resolution", type=int, default=512)
    parser.add_argument("--above", action='store_true')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    raw_dataset_path = root_data / args.raw_dataset_path
    destination_path = root_data / args.destination_path
    target_resolution = args.target_resolution

    c = 0
    for scene_folder_path in list_recursive(raw_dataset_path):
        if not os.path.isdir(scene_folder_path): continue

        transforms = {
            "camera_model": "OPENCV",
            "frames": []
        }

        transforms_opencv = {
            "frames": []
        }
        
        scene_folder_path = Path(scene_folder_path)
        scene_name = scene_folder_path.name
        os.makedirs(os.path.join(destination_path, scene_name, 'rgb'), exist_ok=True)
        for i, rgb_path in enumerate(list_recursive(os.path.join(scene_folder_path, 'rgb'))):
            relative_path = os.path.relpath(rgb_path, scene_folder_path)
            intrinsics_path = os.path.join(scene_folder_path, 'intrinsics.txt')
            pose_path = rgb_path.replace('rgb', 'pose').replace('png', 'txt')
            rgb_path = Path(rgb_path)

            if not os.path.isfile(pose_path):
                print(f"Skipping pose_path {pose_path}")
                continue
            if not os.path.isfile(rgb_path):
                print(f"Skipping rgb_path {rgb_path}")
                continue
            if not os.path.isfile(intrinsics_path):
                print(f"Skipping intrinsics_path {intrinsics_path}")
                continue
            
            with open(pose_path, 'r') as f:
                pose_open_cv = np.array([float(n) for n in f.read().split(' ')]).reshape(4, 4)
                pose_open_gl = open_cv_to_open_gl(pose_open_cv)
                
            with open(intrinsics_path, 'r') as f:
                first_line = f.read().split('\n')[0].split(' ')
                focal = float(first_line[0]) 
                cx = float(first_line[1])
                cy = float(first_line[2])
                            
                orig_img_size = args.og_resolution  # cars_train has intrinsics corresponding to image size of 512 * 512
                img_size = args.target_resolution

            downscale_factor =  orig_img_size / img_size

            if args.above and get_altitude(pose_open_gl) < 0:
                continue

            # Opening and downscaling rgb image
            img = cv2.imread(str(rgb_path))
            img = cv2.resize(img, (img_size, img_size))
            cv2.imwrite(os.path.join(destination_path, scene_name, 'rgb', rgb_path.name), img)
            
            transforms["frames"].append({
                "file_path" : relative_path, 
                "transform_matrix": pose_open_gl.tolist(),
                "fl_x": focal/downscale_factor,
                "fl_y": focal/downscale_factor, 
                "cx": cx/downscale_factor,
                "cy": cy/downscale_factor,
                "w": img_size,
                "h": img_size,
            })

            transforms_opencv["frames"].append({
                "file_path" : relative_path, 
                "transform_matrix": pose_open_cv.tolist(),
                "fl_x": focal/downscale_factor,
                "fl_y": focal/downscale_factor, 
                "cx": cx/downscale_factor,
                "cy": cy/downscale_factor,
                "w": img_size,
                "h": img_size,
            })
    
        with open(os.path.join(destination_path, scene_name, 'transforms.json'), 'w') as outfile:
            json.dump(transforms, outfile, indent=4)


        with open(os.path.join(destination_path, scene_name, 'transforms_opencv.json'), 'w') as outfile:
            json.dump(transforms_opencv, outfile, indent=4)

        if args.debug and c > 4:
            break
        c += 1
