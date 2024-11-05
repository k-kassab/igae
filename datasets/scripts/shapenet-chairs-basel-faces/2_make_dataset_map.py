'''
    This script is used to create an order of the scenes and the dataset while
    excluding the corrupted scenes in exclude.py
    This should be re-run everytime the exluded scenes are changes.
'''

import json
import os
from pathlib import Path
import argparse

ALL_EXCLUDED_SEQUENCES = {}

if __name__ == '__main__':
    dataset_map_fname = "dataset_map.json"
    if os.environ['USER'] == "k.kassab":
        root_data = Path("/home/k.kassab/3da-ae-data")
    elif os.environ['USER'] == "a.schnepf":
        root_data = Path("/home/a.schnepf/phd/data")
    else:
        raise ValueError
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_path", type=str, default="shapenet/chairs/raw")
    parser.add_argument("--destination_path", type=str, default="shapenet/chairs")
    args = parser.parse_args()

    raw_dataset_path = root_data / args.raw_dataset_path
    location_of_dataset_map = root_data / args.destination_path

    categories = ["hats"]
    dataset_map = {}

    for category in categories:
        dataset_map[category] = []
        
        all_scene_ids = os.listdir(os.path.join(raw_dataset_path))
        all_scene_ids.sort()
        for scene_id in all_scene_ids: 

            category_exluded_scenes = ALL_EXCLUDED_SEQUENCES.get(category, [])
            is_excluded = scene_id in category_exluded_scenes
            
            if not is_excluded:
                dataset_map[category].append(scene_id)


    with open(os.path.join(location_of_dataset_map, dataset_map_fname), 'w') as file:
        json.dump(dataset_map, file, indent=4)  

