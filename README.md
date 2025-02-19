# Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder
**Official paper implementation**
> Antoine Schnepf*, Karim Kassab*, Jean-Yves Franceschi, Laurent Caraffa, Flavian Vasile, Jeremie Mary, Andrew Comport, Valérie Gouet-Brunet (* indicates equal contribution)<br>
| [Project Page](https://ig-ae.github.io) | [Full Paper](https://arxiv.org/abs/2410.22936) |<br>

<b>Abstract:</b> *While pre-trained image autoencoders are increasingly utilized in computer vision, the application of inverse graphics in 2D latent spaces has been under-explored. Yet, besides reducing the training and rendering complexity, applying inverse graphics in the latent space enables a valuable interoperability with other latent-based 2D methods. The major challenge is that inverse graphics cannot be directly applied to such image latent spaces because they lack an underlying 3D geometry. In this paper, we propose an Inverse Graphics Autoencoder (IG-AE) that specifically addresses this issue. To this end, we regularize an image autoencoder with 3D-geometry by aligning its latent space with jointly trained latent 3D scenes. We utilize the trained IG-AE to bring NeRFs to the latent space with a latent NeRF training pipeline, which we implement in an open-source extension of the Nerfstudio framework, thereby unlocking latent scene learning for its supported methods. We experimentally confirm that Latent NeRFs trained with IG-AE present an improved quality compared to a standard autoencoder, all while exhibiting training and rendering accelerations with respect to NeRFs trained in the image space.*

![LatentScenes](assets/latent_scenes.gif)

# Paper implementation details
Our paper utilizes two codebases:
- The [first](./igae_training) is utilized to train an IG-AE.
- The [second](./latent-nerfstudio) is a dedicated subrepo that extends [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) to support training various NeRF architectures in the latent space. It is utilized in our paper to train our NeRF models in the latent space of a standard AE as well as that of our IG-AE.

## License

Each codebase has its own license. For more information, please refer to each of the codebases.

## Citation

If you find this research project useful, please consider citing our work:
```
@inproceedings{
  ig-ae,
  title={Bringing Ne{RF}s to the Latent Space: Inverse Graphics Autoencoder},
  author={Antoine Schnepf and Karim Kassab and Jean-Yves Franceschi and Laurent Caraffa and Flavian Vasile and Jeremie Mary and Andrew I. Comport and Valerie Gouet-Brunet},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=LTDtjrv02Y}
}
```
