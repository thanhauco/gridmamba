import torch

from gridmamba.ssm import MambaBlock, parallel_scan, selective_scan, sequential_scan


def test_parallel_scan_matches_sequential():
    torch.manual_seed(0)
    for length in (1, 2, 7, 28, 33):
        a = torch.rand(3, length, 5, 4)
        b = torch.randn(3, length, 5, 4)
        torch.testing.assert_close(parallel_scan(a, b), sequential_scan(a, b), rtol=1e-5, atol=1e-5)


def test_selective_scan_backends_agree_with_gradients():
    torch.manual_seed(1)
    u = torch.randn(2, 16, 8, requires_grad=True)
    delta = torch.rand(2, 16, 8) * 0.1
    A = -torch.rand(8, 4)
    B, C = torch.randn(2, 16, 4), torch.randn(2, 16, 4)
    D = torch.ones(8)
    y_par = selective_scan(u, delta, A, B, C, D, parallel=True)
    y_seq = selective_scan(u, delta, A, B, C, D, parallel=False)
    torch.testing.assert_close(y_par, y_seq, rtol=1e-5, atol=1e-5)
    (g_par,) = torch.autograd.grad(y_par.sum(), u)
    (g_seq,) = torch.autograd.grad(y_seq.sum(), u)
    torch.testing.assert_close(g_par, g_seq, rtol=1e-4, atol=1e-5)


def test_mamba_block_is_causal():
    torch.manual_seed(2)
    block = MambaBlock(d_model=16).eval()
    x = torch.randn(1, 12, 16)
    y = block(x)
    x2 = x.clone()
    x2[:, 8:] += torch.randn(1, 4, 16)
    y2 = block(x2)
    torch.testing.assert_close(y[:, :8], y2[:, :8])
    assert not torch.allclose(y[:, 8:], y2[:, 8:])
