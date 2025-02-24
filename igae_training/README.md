# IGAE Training
> Antoine Schnepf*, Karim Kassab*, Jean-Yves Franceschi, Laurent Caraffa, Flavian Vasile, Jeremie Mary, Andrew Comport $^\dagger$, Valérie Gouet-Brunet $^\dagger$ (*$^\dagger$ indicate equal contributions)<br>
| [Project Page](https://ig-ae.github.io) | [Full paper](https://openreview.net/forum?id=LTDtjrv02Y) | [Preprint](https://arxiv.org/abs/2410.22936) |<br>

This section is dedicated to training an Inverse Graphics Autoencoder.

## Setup
In this section we detail how prepare the environment for training IG-AE.

### Environment 
Our code has been tested on:
- Linux (Debian)
- Python 3.11.9
- Pytorch 2.0.1
- CUDA 11.8
- `L4` and `A100` NVIDIA GPUs


You can use Anaconda to create the environment:
```
conda create --name igae -y python=3.11.9
conda activate igae
```
Then, you can install pytorch with Cuda 11.8 using the following command:
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```
_You may have to adapt the cuda version according to your hardware, we recommend using CUDA >= 11.8_

To install the remaining requirements, execute:
```
pip install -r requirements.txt
```

## Usage

### Download data
The data to train IG-AE is available at our [Hugging Face repository](https://huggingface.co/datasets/k-kassab/igae-data/tree/main).

To download the data, make sure you have [git lfs](https://git-lfs.com) installed:
```
      apt install git-lfs
      git lfs install
```
and then clone our dataset repository, by running:
```
      git clone https://huggingface.co/datasets/k-kassab/igae-data
```
Note that the data should take around 45 GB of disk space.

### Define data directory
You must specify the path to the igae-data by defining the environment variable DATA_DIR
```
export DATA_DIR=".../igae-data"
```
or by changing the variable ``DATA_DIR`` in datasets/dataset.py .

### Train IG-AE
To train IG-AE, run:
```
bash run.sh igae.yaml
```

## Visualization / evaluation
We visualize and evaluate our method using [wandb](https://wandb.ai/site). 
You can get quickstarted [here](https://docs.wandb.ai/quickstart).

## Notice
This repository additionally implements components present in our other work "[Scaled Inverse Graphics: Efficiently Learning Large Sets of 3D Scenes](https://scaled-ig.github.io)" that are not used for this paper. 
This mainly includes the Micro-Macro Tri-Planes decomposition.
Feel free to explore our other work if it piques your interest. 

## A Note on License

This code is open-source. We share most of it under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
However, we reuse code from [EG3D](https://github.com/NVlabs/eg3d) which is released under a more restrictive [license](ae/volume_rendering/LICENSE.txt) that requires redistribution under the same license or equivalent. 
Hence, the corresponding parts of our code ([ray_marcher.py](ae/volume_rendering/ray_marcher.py), [ray_sampler.py](ae/volume_rendering/ray_sampler.py), [renderer.py](ae/volume_rendering/renderer.py), [triplane_renderer.py](ae/triplane_renderer.py) and [camera_utils.py](ae/camera_utils.py)) are open-sourced using the [original license](https://github.com/NVlabs/eg3d/blob/main/LICENSE.txt) of these works and not Apache. 

## Citation

If you find this research project useful, please consider citing our work:
```
@inproceedings{
      ig-ae,
      title={{Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder}},
      author={Antoine Schnepf and Karim Kassab and Jean-Yves Franceschi and Laurent Caraffa and Flavian Vasile and Jeremie Mary and Andrew I. Comport and Valerie Gouet-Brunet},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=LTDtjrv02Y}
}
```
