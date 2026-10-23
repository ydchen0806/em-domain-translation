# Energy-Guided Latent Flow Translation for 3D EM Segmentation

Research-code snapshot accompanying the ISBI manuscript **Energy-Guided Diffusion for Domain-Adaptive 3D Neuron Segmentation**.

## Method overview

A target-domain latent flow model is trained with direct clean-latent prediction and a velocity-space L1 loss. Source-conditioned translation combines classifier-free guidance and two source-referenced sampling energies:

- **SCG:** squared distance between the predicted and source-reference latents after 3D average pooling (kernel 3, stride 1, padding 1).
- **DAG:** channelwise cosine similarity of the predicted and noisy source-reference latents.

The translated volume is paired with the source voxel-neighbor affinity labels to train an affinity-predicting 3D residual U-Net. Post-processing converts affinities into neuron instances.

## Code map

- `networks_copy/denoiser.py`: direct clean-latent flow matching, sampling and SCG/DAG energies.
- `sampling.py`: source-to-target translation CLI (`--lambdalist DAG SCG`).
- `scripts/train_model.py`: target-domain flow and affinity-conditioned ControlNet training entry point.
- `scripts/seg_train_maisi.py`: affinity training and target consistency.
- `data/dataset_lb.py`, `data/dataset_unlb.py`: dataset and supervision handling.
- `configs/`: model, training, and inference configuration snapshots.

## Datasets and setup

This project studies 3D electron microscopy datasets CREMI-C and AC4. **Data, labels, pretrained VQGAN weights, flow model weights, and trained ControlNet weights are not bundled.** Configure paths in `configs/` before running. Example paths were converted to `./local_data/...` in this artifact and are not included in the repository.

A CUDA-capable PyTorch environment with matching MONAI, Waterz and graph segmentation dependencies is required. `requirements_original.txt` captures the originally supplied partial environment; `requirements.txt` lists additional observed Python packages but does **not** constitute a tested, pinned reproduction environment.

For translation, the CLI parses weights explicitly:

```bash
python sampling.py --help
# supply the actual trained checkpoints and datasets in configs/
# example only, not a claim about the paper's experiment weights:
python sampling.py --lambdalist 0.0 0.0
```

**Reproducibility note:** This is a cleaned research snapshot, not an independently reproduced training release. Several entry-point/configuration mismatches remain in the original supplied code. The reported results and final experimental command lines require comparison with actual run logs before claiming full reproduction. See `CODE_AUDIT.md` for details.

## License and upstream dependencies

The source combines components influenced by or derived from several upstream systems (including MONAI/MAISI and standard connectomics code). Original source attribution and the supplied `LICENSE.weights` file have been preserved. Redistribution, licensing compatibility and availability of pretrained weights must be confirmed by the copyright holders before public release. No new license is assigned to material with unresolved third-party provenance.

This repository is intended as an **anonymized reviewer artifact**. It includes no original raw datasets or experimental checkpoints.