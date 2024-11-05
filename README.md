# Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder
**Official paper implementation**
> Antoine Schnepf*, Karim Kassab*, Jean-Yves Franceschi, Laurent Caraffa, Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet (* indicates equal contribution)<br>
| [Project Page](https://ig-ae.github.io) | [Full Paper](https://arxiv.org/abs/2410.22936) |<br>

<b>Abstract:</b> *While pre-trained image autoencoders are increasingly utilized in computer vision, the application of inverse graphics in 2D latent spaces has been under-explored. Yet, besides reducing the training and rendering complexity, applying inverse graphics in the latent space enables a valuable interoperability with other latent-based 2D methods. The major challenge is that inverse graphics cannot be directly applied to such image latent spaces because they lack an underlying 3D geometry. In this paper, we propose an Inverse Graphics Autoencoder (IG-AE) that specifically addresses this issue. To this end, we regularize an image autoencoder with 3D-geometry by aligning its latent space with jointly trained latent 3D scenes. We utilize the trained IG-AE to bring NeRFs to the latent space with a latent NeRF training pipeline, which we implement in an open-source extension of the Nerfstudio framework, thereby unlocking latent scene learning for its supported methods. We experimentally confirm that Latent NeRFs trained with IG-AE present an improved quality compared to a standard autoencoder, all while exhibiting training and rendering accelerations with respect to NeRFs trained in the image space.*

![LatentScenes](assets/latent_scenes.gif)

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
(coming soon)
Download and untar the data (about 45 GB).

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

## Citation

If you find this research project useful, please consider citing our work:
```
@article{ig-ae,
      title={{Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder}}, 
      author={Antoine Schnepf and Karim Kassab and Jean-Yves Franceschi and Laurent Caraffa and Flavian Vasile and Jeremie Mary and Andrew Comport and Valérie Gouet-Brunet},
      journal={arXiv preprint arXiv:2410.22936},
      year={2024}
}
```