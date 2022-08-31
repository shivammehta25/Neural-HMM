"""functions.py.

File for custom utility functions to improve numerical precision
"""
import torch


def log_clamped(x, eps=1e-04):
    clamped_x = torch.clamp(x, min=eps)
    return torch.log(clamped_x)


def inverse_sigmod(x):
    r"""
    Inverse of the sigmoid function
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return log_clamped(x / (1.0 - x))


def inverse_softplus(x):
    r"""
    Inverse of the softplus function
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return log_clamped(torch.exp(x) - 1.0)


def logsumexp(x, dim):
    r"""
    Differentiable LogSumExp: Does not creates nan gradients
        when all the inputs are -inf
    Args:
        x : torch.Tensor -  The input tensor
        dim: int - The dimension on which the log sum exp has to be applied
    """

    m, _ = x.max(dim=dim)
    mask = m == -float("inf")

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


def log_domain_matmul(log_a, log_b):
    r"""
    Multiply two matrices in log domain
    Args:
        log_a : m x n
        lob_b : n x p
        out : m x p
    Returns:
        Computes output_{i, j} = logsumexp_k [ log_A_{i, k} + log_B{k, j} ]
    """

    m, n, p = log_a.shape[0], log_a.shape[1], log_b.shape[1]

    # Dimensions must be same to add

    # Expand A to the p size
    log_A_expanded = log_a.unsqueeze(2).expand((m, n, p))
    # Expand B to m size
    log_B_expanded = log_b.unsqueeze(0).expand((m, n, p))
    # These expansion will result in addition

    elementwise_sum = log_A_expanded + log_B_expanded

    out = logsumexp(elementwise_sum, 1)

    return out


def masked_softmax(vec, dim=0):
    r"""Outputs masked softmax"""
    mask = ~torch.eq(vec, 0)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    softmax_values = masked_exps / masked_sums
    return softmax_values


def masked_log_softmax(vec, dim=0):
    r"""Outputs masked log_softmax"""
    mask = ~torch.eq(vec, 0)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    softmax_values = masked_exps / masked_sums
    idx = softmax_values != 0
    softmax_values[idx] = torch.log(softmax_values[idx])
    return softmax_values


def get_mask_from_len(lengths, device="cpu", out_tensor=None):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=device) if out_tensor is None else torch.arange(0, max_len, out=out_tensor)
    mask = ids < lengths.unsqueeze(1)
    return mask


def get_mask_for_last_item(lengths, device="cpu", out_tensor=None):
    """Returns n-1 mask for the last item in the sequence.

    Args:
        lengths (torch.IntTensor): lengths in a batch
        device (str, optional): Defaults to "cpu".
        out_tensor (torch.Tensor, optional): uses the memory of a specific tensor.
            Defaults to None.
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=device) if out_tensor is None else torch.arange(0, max_len, out=out_tensor)
    mask = ids == lengths.unsqueeze(1) - 1
    return mask
