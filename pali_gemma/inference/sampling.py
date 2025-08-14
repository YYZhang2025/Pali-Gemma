import torch


def _naive_sample(next_token_logits: torch.Tensor, **kwargs):
    return torch.argmax(next_token_logits, dim=-1, keepdim=True)


def _sample_top_k(next_token_logits: torch.Tensor, k: int = 50, temperature: float = 1.0, **kwargs):
    probs = torch.softmax(next_token_logits / temperature, dim=-1)
    probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True)

    probs_sort = probs_sort[:, :k]
    probs_indices = probs_indices[:, :k]

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # Normalize

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = probs_indices.gather(-1, next_token)

    return next_token


def _sample_top_p(next_token_logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0, **kwargs):
    probs = torch.softmax(next_token_logits / temperature, dim=-1)
    probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)

    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # Normalize

    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = probs_indices.gather(-1, next_token)

    return next_token


def get_sampler(sampling_method: str):
    if sampling_method == "naive":
        return _naive_sample
    elif sampling_method == "top_k":
        return _sample_top_k
    elif sampling_method == "top_p":
        return _sample_top_p
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
