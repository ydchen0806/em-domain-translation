"""Minimal CPU test for joint SCG/DAG gradient (requires torch)."""
import torch
import torch.nn.functional as F


def test_joint_gradient_matches_separate_gradients():
    torch.manual_seed(17)
    z = torch.randn((1, 4, 4, 5, 6), requires_grad=True)
    reference = torch.randn_like(z)
    # Differentiable shared feature transformation like the denoiser branch.
    prediction = torch.nn.functional.silu(1.4 * z + 0.2)
    style = F.cosine_similarity(prediction, reference, dim=1).mean(dim=(1, 2, 3))
    pooled_pred = F.avg_pool3d(prediction, 3, stride=1, padding=1)
    pooled_ref = F.avg_pool3d(reference, 3, stride=1, padding=1)
    structure = ((pooled_pred - pooled_ref) ** 2).sum(dim=(1, 2, 3, 4))
    style_weight, structure_weight = 200.0, 1.0
    expected_style = torch.autograd.grad(style.sum(), z, retain_graph=True)[0]
    expected_structure = torch.autograd.grad(structure.sum(), z, retain_graph=True)[0]
    actual = torch.autograd.grad((style_weight * style + structure_weight * structure).sum(), z)[0]
    assert torch.allclose(actual, style_weight * expected_style + structure_weight * expected_structure, rtol=1e-5, atol=1e-4)
