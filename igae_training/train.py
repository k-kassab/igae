# Copyright 2024 Antoine Schnepf, Karim Kassab, Jean-Yves Franceschi, Laurent Caraffa, 
# Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from prodict import Prodict
import argparse
import sys
import pickle
import itertools
import math
from ae.utils import load_config, remove_empty_categories
from ae.normalizer import TanhNormalizer
from ae.trainers import NerfVAE, Trainer, init_models, init_datasets
from ae.evaluator import Evaluator
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


# ----------------------------------------------------------------------        
def check_injection_data(dataset_train, dataset_injection, t_args) : 
    len_dataloader = math.floor(len(dataset_train) / t_args.consistency.batch_size)
    len_injection_dataloader = math.floor(len(dataset_injection) / t_args.injection.batch_size)
    error_msg = f"Not enough injection data. Can do {len_dataloader} consistency iter but only {len_injection_dataloader} injection iter, and they shloud be done injection every {t_args.injection.every} consistency iterations"
    assert math.floor(len_dataloader / t_args.injection.every) <= math.floor(len_injection_dataloader), error_msg


if __name__ == '__main__' :
    debug = False

    REPO_PATH=os.path.dirname(os.path.realpath(__file__))

    # --- args parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="train_default.yaml")
    parser.add_argument('--config_dir', type=str, default=os.path.join(REPO_PATH, "configs"))
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--train', action="store_true")
    
    if debug:
        print("/!!!!\ Debug mode /!!!!\ ")
        args = parser.parse_args([
            "--pretrain",
            "--config", "igae_debug.yaml",
        ])
    else:
        args = parser.parse_args()
    
    
    # -- fetch configuration --
    config = Prodict.from_dict(load_config(args.config, args.config_dir, from_default=True, default_cfg_name="train_default.yaml"))
    config = remove_empty_categories(config)
    assert config.pretrain_exp_name != config.exp_name, "pretrain_exp_name and exp_name must be different"
    
    # -- init train or pretrain --
    assert (args.pretrain != args.train), "Either pretrain or train must be True in script args" #(XOR) one of the two must be true

    do_pretraining = args.pretrain and config.pretrain
    do_training = args.train and config.train

    if do_pretraining:
        t_args = config.pretrain_args
        expname = config.pretrain_exp_name
    elif do_training:
        t_args = config.train_args
        expname = config.exp_name

    assert do_pretraining != do_training, "Either pretrain or train must be True in config" #(XOR) one of the two must be true
    
    if args.pretrain and not config.pretrain:
        print(f"Skipping pretrain run (config.pretrain={config.pretrain}).")
        print("Exiting.")
        sys.exit()

    # -- intialising accelerator --
    accelerator_kwargs = {'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters=True)]}
    accelerator = Accelerator(**accelerator_kwargs)
    
    # --- initialiasing data ---
    train_scenes, multi_scene_trainset, train_injection_dataset, test_scenes, multi_scene_testset, test_injection_dataset, pose_sampler, num_scenes = init_datasets(config, include_injection=t_args.injection.apply)

    # --- initialiasing models ---
    vae, latent_renderer, latent_nerfs = init_models(config, num_scenes)    

    evaluator = Evaluator(train_scenes, test_scenes, train_injection_dataset, test_injection_dataset, pose_sampler, config, REPO_PATH)
    normalizer = TanhNormalizer(scale=config.vae.normalization.scale, eps=config.vae.normalization.eps)
    nerf_vae = NerfVAE(            
        vae,
        latent_renderer,
        latent_nerfs,
        normalizer
    )

    # load model in case of training
    if do_training:
        # load the latent nerfs and the latent renderer
        if accelerator.is_main_process:
            load_path = os.path.join(REPO_PATH, config.savedir, config.pretrain_exp_name)
            checkpoint = torch.load(
                os.path.join(load_path, "gvae_latest.pt"),
                map_location=torch.device('cpu')
            )
            nerf_vae.vae.load_state_dict(checkpoint['vae'])
            nerf_vae.latent_renderer.load_state_dict(checkpoint['renderer'])
            nerf_vae.latent_nerfs.load_state_dict(checkpoint['nerfs'])


    if t_args.injection.apply:
        check_injection_data(multi_scene_trainset, train_injection_dataset, t_args)

    trainer = Trainer(
        config=config, 
        t_args=t_args,
        expname=expname, 
        accelerator=accelerator,
        multi_scene_set=multi_scene_trainset,
        injection_set=train_injection_dataset,
        nerf_vae=nerf_vae, 
        normalizer=normalizer, 
        evaluator=evaluator, 
        repo_path=REPO_PATH,
        debug=False
    )
    trainer.train()

    if accelerator.is_local_main_process:
        os.system(f"rm -r {os.path.join(REPO_PATH, config.savedir, config.exp_name, 'buffer', '*')}")

    accelerator.end_training()
