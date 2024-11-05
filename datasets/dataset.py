from pathlib import Path
from typing import Union, List
import PIL.Image
import json
import os 
from prodict import Prodict
import torchvision.transforms as tr
import numpy as np
import torch
import math
import string
import random
import tqdm
import itertools

if "DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["DATA_DIR"])
else:
    DATA_DIR = Path("enter/path/to/data/dir")

def frame_to_datapoint(frame):
    relative_path = frame['file_path']
    extrinsics = torch.Tensor(frame['transform_matrix'])
    intrinsics = torch.eye(3)
    intrinsics[0][0] = frame['fl_x']/frame['w']
    intrinsics[1][1] = frame['fl_y']/frame['h']
    intrinsics[0][2] = frame['cx']/frame['w']
    intrinsics[1][2] = frame['cy']/frame['h']

    return (relative_path, torch.concatenate((extrinsics.flatten(), intrinsics.flatten())))

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in [".png", ".jpeg", ".jpg"] # type: ignore

class DataReader:
    "This class allows to read all the scenes and their views from the disk via a dataset_map"

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.dataset_path = Path(DATA_DIR) / dataset_config.name / dataset_config.subset
        self.few_view_factor = dataset_config.few_view_factor

        # 0. Load dataset_map 
        with open(DATA_DIR / dataset_config.name / "dataset_map.json", 'r') as f:
            self.dataset_map = json.load(f)

        # 1. extract the scenes from the dataset_map and the dataset_config
        self.path_of_scenes = []
        for scene_category, scene_range in dataset_config.scene_repartition.items():
            all_possible_scenes = self.dataset_map[scene_category]
            if scene_range[1] > len(all_possible_scenes):
                print(f"WARNING: {scene_category}: {scene_range[1]} is larger than the number of scenes in the dataset ({len(all_possible_scenes)}).")
                print(f"Taking {len(all_possible_scenes)} instead.")
            for i in range(scene_range[0], min(scene_range[1], len(all_possible_scenes))):
                self.path_of_scenes.append(
                        self.dataset_path / all_possible_scenes[i]
                )

        # 2. extract the data for each scene
        self.data_per_scene = {}

        # iterate over scenes
        for scene_path in self.path_of_scenes:

            # load transforms.json
            with open(os.path.join(scene_path, "transforms_opencv.json"), 'r') as f:
                transforms = json.load(f)
            
            # load the images and corresponding poses
            self.data_per_scene[scene_path] = []
            for frame in transforms['frames']:
                datapoint = frame_to_datapoint(frame)
                self.data_per_scene[scene_path].append(datapoint)
            if self.few_view_factor < 1.0:
                randomness = random.Random(1234)
                selected_data_per_scene = randomness.sample(self.data_per_scene[scene_path], int(len(self.data_per_scene[scene_path]) * self.few_view_factor))
                self.data_per_scene[scene_path] = selected_data_per_scene
                print(f"WARNING: Using only a fraction ({self.few_view_factor*100}%) of the views for each scene.")

        # 4. init the transforms
        mu=0.5
        sigma=0.5
        self.transform = tr.Compose([
            tr.ToTensor(),
            tr.Normalize([mu], [sigma])
        ])
        self.inv_transform = tr.Compose([
            tr.Normalize([-mu/sigma], [1/sigma]),
            tr.ToPILImage()
        ])

    
    def get_image(self, idx_obj:int, idx_img:int) : 
        scene_path = self.path_of_scenes[idx_obj]
        img_rel_path, pose = self.data_per_scene[scene_path][idx_img]
        img =  self.transform(PIL.Image.open(scene_path / img_rel_path).convert('RGB'))

        return dict(img=img, pose=pose)

class ObjectDataset(torch.utils.data.Dataset):
    "A dataset class for a single object / scene, allowing to specify a certain split on a scen" 

    def __init__(self, data_reader, selected_obj_idx:int=0, split='all', split_ratio=0.9):
        """
        args:
            selected_obj_idx: int: index of the object to be selected
            split: str: 'all' or 'train' or 'test'. 
        """

        # 0. Init the data reader
        self.data_reader = data_reader
        self.inv_transform = data_reader.inv_transform

        # 1. Select the object correponding to the index 
        assert selected_obj_idx < len(self.data_reader.path_of_scenes)
        self.selected_obj_idx = selected_obj_idx
        self.selected_obj_id = self.data_reader.path_of_scenes[self.selected_obj_idx]

        # 2. Init the split
        self.split = split
        self.n_img = len(self.data_reader.data_per_scene[self.selected_obj_id])
        self.n_train_img = math.ceil(self.n_img * split_ratio)
        self.n_test_img = self.n_img  - self.n_train_img
        if self.n_test_img <= 0:
            print(f"WARNING: Not enough images for the test split for object {self.selected_obj_id}, got {self.n_test_img} test images from {self.n_img} images.")


    def __len__(self):
        if self.split == 'all':
            return self.n_img
        elif self.split == 'train':
            return self.n_train_img
        elif self.split == 'test':
            return self.n_test_img
        else:
            raise ValueError(f"Unknown split {self.split}")

    def _get_idx_(self, idx):
        if self.split == 'all':
            idx_ = idx
        elif self.split == 'train':
            if not(0 <= idx < len(self)):
                raise ValueError(f"Index {idx} is out of bounds for split '{self.split}'")
            idx_ = idx + self.n_test_img
        elif self.split == 'test':
            if not(0 <= idx < self.n_test_img):
                raise ValueError(f"Index {idx} is out of bounds for split '{self.split}'")
            idx_ = idx
        else:
            raise ValueError(f"Unknown split {self.split}")
        return idx_


    def __getitem__(self, idx):
        idx_ = self._get_idx_(idx)
        return self.data_reader.get_image(self.selected_obj_idx, idx_)

class CachedDataset(torch.utils.data.Dataset) : 

    def __init__(self, dataset, vae, device, batchsize, use_mean, repo_path, savedir, exp_name) : 
        self.dataset = dataset

        rd_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self.save_dir = os.path.join(repo_path, savedir, exp_name, 'buffer', f"CachedDataset_{rd_string}")

        if not os.path.exists(self.save_dir) : 
            os.makedirs(self.save_dir)

        self._process_source_dataset(vae, batchsize, num_workers=4, device=device, use_mean=use_mean)

    @torch.no_grad()
    def _process_source_dataset(self, vae, batch_size, num_workers, device, use_mean=False):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        old_device = vae.device
        vae.to(device)

        running_idx = 0
        for i, batch in enumerate(tqdm.tqdm(dataloader, desc="Caching latents")) :
            img = batch['img'].to(device)
            latent_dist = vae.encode(img).latent_dist

            if use_mean: 
                latent_imgs = latent_dist.mean.cpu()
            else:
                latent_imgs = latent_dist.sample().cpu()

            for latent_img in latent_imgs:
                torch.save(latent_img, os.path.join(self.save_dir, f"latent_{running_idx}.pt"))
                running_idx += 1

        vae.to(old_device)
    
    def get_latents(self, idx) : 
        return torch.load(os.path.join(self.save_dir, f"latent_{idx}.pt"), map_location='cpu')

    def __len__(self) :
        return len(self.dataset)
    
    def __getitem__(self, idx) : 
        res =  self.dataset[idx]
        res.update({
            'cached_latent' : self.get_latents(idx)
        })
        return res
    
class MultiSceneDataset(torch.utils.data.Dataset):
    "This class aggregates all objects together for common indexing"
    def __init__(self, datasets: List[ObjectDataset]):
        self.datasets = datasets
        self.n_scenes = len(datasets)

        self.n_img_per_scene = np.array([len(dataset) for dataset in datasets])
        self.cumlen = np.cumsum(self.n_img_per_scene)
        self.padcumlen = np.concatenate([[0], self.cumlen[:-1]])
        self.inv_transform = datasets[0].inv_transform
        self.return_category = False
        self.custom_transform = None

    def _get_scene_and_obj_idx(self, idx):
        scene_idx = np.searchsorted(self.cumlen, idx + 1)
        pad = self.padcumlen[scene_idx]
        obj_idx = idx - pad
        return scene_idx, obj_idx
    
    def _get_obj_category_from_path(self, scene_path):
        obj_name = scene_path.name
        dataset_map_path = scene_path.parent.parent / "dataset_map.json"
        with open(dataset_map_path, "r") as f:
            dataset_map = json.load(f)
        for key, values in dataset_map.items():
            if obj_name in values:
                return key
        raise ValueError(f"Object {obj_name} not found in dataset_map.json")

    def _get_obj_category(self, scene_idx):
        current_dataset = self.datasets[scene_idx]
        scene_path = current_dataset.data_reader.path_of_scenes[current_dataset.selected_obj_idx]
        return self._get_obj_category_from_path(scene_path)
    
    def with_transform(self, custom_transform):
        self.custom_transform = custom_transform
        return self

    def __getitem__(self, idx):
        scene_idx, obj_idx = self._get_scene_and_obj_idx(idx)
        res = self.datasets[scene_idx][obj_idx]
        res.update({'scene_idx': scene_idx})
        if self.return_category:
            res.update({'category': self._get_obj_category(scene_idx)})
        res = self.custom_transform(res) if self.custom_transform is not None else res
        return res

    def __len__(self):
        return self.n_img_per_scene.sum()


class DatasetIntrinsics:
    def __init__(
            self,
            name, 
            focal,
            cx, cy,
            azimuth_range,
            elevation_range,
            camera_distance,
        ) :
        self.name = name
        self.focal = focal
        self.cx = cx
        self.cy = cy
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.camera_distance = camera_distance

class DatasetIntrinsicsManager:

    intrinsics = {
        'objaverse/small' : DatasetIntrinsics(
            name='objaverse',
            focal=1.09375,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'objaverse/small_curated' : DatasetIntrinsics(
            name='objaverse',
            focal=1.09375,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'objaverse/medium' : DatasetIntrinsics(
            name='objaverse',
            focal=1.09375,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'objaverse/medium_curated' : DatasetIntrinsics(
            name='objaverse',
            focal=1.09375,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'objaverse/medium_manually_curated' : DatasetIntrinsics(
            name='objaverse',
            focal=1.09375,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'objaverse/medium_many_views_manually_curated' : DatasetIntrinsics(
            name='objaverse',
            focal=1.09375,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'shapenet/all' : DatasetIntrinsics(
            name='shapenet',
            focal=1.02,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'shapenet/small' : DatasetIntrinsics(
            name='shapenet',
            focal=1.02,
            cx=0.5, cy=0.5, 
            camera_distance=1.3,
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'co3d/small' : DatasetIntrinsics(
            name='co3d_small',
            focal=1.5,  # this actually varies accross scenes, but is only used to generate the eval videos of the scenes, so we can set it to a fixed value
            cx=0.5, cy=0.5,
            camera_distance=1.5, # this actually varies accross scenes, but is only used to generate the eval videos of the scenes, so we can set it to a fixed value
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),

        'co3d/medium' : DatasetIntrinsics(
            name='co3d_small',
            focal=1.5,  # this actually varies accross scenes, but is only used to generate the eval videos of the scenes, so we can set it to a fixed value
            cx=0.5, cy=0.5,
            camera_distance=1.5, # this actually varies accross scenes, but is only used to generate the eval videos of the scenes, so we can set it to a fixed value
            azimuth_range=[-math.pi, math.pi],
            elevation_range=[-math.pi/2, math.pi/2],
        ),
    }
    


    @staticmethod
    def get_intrinsics(dataset_name):
        return DatasetIntrinsicsManager.intrinsics[dataset_name]

class ImageDataset(torch.utils.data.Dataset) : 
    "This class reads all the images in a folder and makes them available as a dataset."

    def __init__(self, dataset_config, split='all', seed=1234) : 
        assert split in ['all', 'train', 'test']
        
        # 1. Init the image paths
        self.dataset_path = Path(DATA_DIR) / dataset_config.name 
        all_path = sorted(self.dataset_path.rglob('*'))
        all_path = [f for f in all_path if is_image_ext(f) and os.path.isfile(f)] 

        randomness = random.Random(seed)
        if len(all_path) < dataset_config.max_img:
            print(f"WARNING: Not enough images in the dataset. Only {len(all_path)} available but {dataset_config.max_img} requested.")
        self.img_paths = randomness.sample(all_path, min(dataset_config.max_img, len(all_path)))
        self.return_category = False

        # 2. Init the split
        self.split = split
        self.n_img = len(self.img_paths)
        self.n_train_img = math.ceil(self.n_img * dataset_config.split_ratio)
        self.n_test_img = self.n_img  - self.n_train_img


        #3. Init the transforms
        mu=0.5
        sigma=0.5
        self.transform = tr.Compose([
            tr.ToTensor(),
            tr.Normalize([mu], [sigma])
        ])
        self.inv_transform = tr.Compose([
            tr.Normalize([-mu/sigma], [1/sigma]),
            tr.ToPILImage()
        ])

    def _get_category(self, idx_):
        category_id = self.img_paths[idx_].parent.name
        # open txt file
        category_map_path = self.dataset_path.parent.parent / "LOC_synset_mapping.txt"
        category_name = None
        with open(category_map_path, 'r') as f:
            for line in f:
                if line.startswith(category_id):
                    category_name = random.choice(line.split(' ', 1)[1].split(', '))
                    category_name = category_name.replace('\n', '')
                    break
        if category_name is None:
            print("WARNING: Category name not found for category id", category_id)
            print("Putting default category instead.")
            category_name = "random photo"
        return category_name


    def __len__(self) : 
        if self.split == 'all':
            return self.n_img
        elif self.split == 'train':
            return self.n_train_img
        elif self.split == 'test':
            return self.n_test_img
        else:
            raise ValueError(f"Unknown split {self.split}")

    def __getitem__(self, idx) : 

        if self.split == 'all':
            idx_ = idx

        elif self.split == 'train':
            if not(0 <= idx < len(self)):
                raise ValueError(f"Index {idx} is out of bounds for split '{self.split}'")
            idx_ = idx + self.n_test_img

        elif self.split == 'test':
            if not(0 <= idx < self.n_test_img):
                raise ValueError(f"Index {idx} is out of bounds for split '{self.split}'")
            idx_ = idx

        else:
            raise ValueError(f"Unknown split {self.split}")

        res = {}
        img = self.transform(PIL.Image.open(self.img_paths[idx_]).convert('RGB'))
        res['img'] = img
        if self.return_category:
            res['category'] = self._get_category(idx_)

        return res
        
class MyIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)
    
    
