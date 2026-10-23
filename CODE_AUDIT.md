# Code-to-paper alignment and reproduction checks

## Verified source mapping

1. `Denoiser.forward()` interpolates clean **target-domain** VQGAN latents and Gaussian noise at sigmoid-normal times. It predicts clean latents and minimizes **velocity-space L1**, not an L2 reconstruction loss.
2. Translation initializes from a noisy **source** latent and advances using Heun integration; guidance uses a separately noised source reference.
3. SCG is **sum of squared errors** between local average-pooled 3D latents; DAG is **channelwise cosine similarity**, spatially averaged. The input order in `--lambdalist` is `[DAG, SCG]` (style, structure), while the paper writes λs for SCG and λd for DAG.
4. The target-domain learned generative model defines the target prior; DAG itself does **not** explicitly compare to a target feature.
5. Segmentation predicts three affinity channels, supervised by weighted **MSE**, with source, synthesized and EMA consistency branches. It does not use ordinary voxelwise cross entropy.
6. `sampling.py` defaults to `t=0.875`, `steps=50`, `cfg=2.9`, and historically `--lambdalist=[0,0]`. The last was fixed to numeric two-argument parsing in this cleaned snapshot.

## Release blockers to resolve before claiming reproducibility

**Potential target-label leakage (critical)**: `scripts/train_model.py` in mode `train_cn` instantiates `SingleDataset_lb` on `args.target_path`; `data/dataset_lb.py` explicitly opens `cremiC_labels.h5` or `AC4_labels.h5` to build affinity controls. If this checkpoint was used for main UDA numbers, target labels entered training and the benchmark is not label-free adaptation. Establish whether the reported experimental checkpoint came from a different source-affinity ControlNet pipeline. Publish the actual run configs and scripts after resolution.

**Original gradient graph error (patched in this snapshot)**: When both energy weights are nonzero, the supplied `_forward_sample` differentiated `energy1` and `energy2` through the same conditional network without retaining the graph. The cleaned copy instead computes a gradient of `lambda_style*energy1 + lambda_struct*energy2` once, which is mathematically equivalent to adding the two gradients and avoids a second backward traversal.

**Original training entry-point mismatch (not patched)**: In `scripts/train_model.py`, `train_one_epoch` is called with extra positional and named arguments not accepted by its local signature; the dataset constructor has also been called with unsupported `norm2`. Other local entry points have similar inconsistencies. Do not claim the supplied entry point was executed successfully without reconciling versions.

**Weights unverified**: The manuscript reports λs=1, λd=200; no experiment log was provided to establish that the result table was generated using those values. Original CLI default was `[0,0]`. Verify run commands and model checkpoints before submission.

**Checkpoint/data unavailability**: The input archive lacks raw volumes, labels, checkpoint tensors, precise GPU/environment versions, and complete experiment logs. The source package is a code release candidate, not a benchmark that was rerun.

**Third-party license review**: inspect original software/code licenses and the restrictions in `LICENSE.weights` before public publishing; preserve copyright and attribution notices.

**Publication note**: ISBI 2027 operates single-blind review; an anonymous code URL is optional, not a substitute for a genuine accessible repository. Add the URL only after the anonymous mirror is published and verified.
