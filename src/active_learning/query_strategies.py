import torch
import torch.nn.functional as F


def mention_detection_entropy(mention_logits: torch.Tensor, threshold=0.0) -> torch.Tensor:
    """See https://aclanthology.org/2022.acl-long.519/ for details

    Args:
        mention_logits (torch.Tensor): An 1d tensor in which each element is a mention score (logit value)

    Returns:
        torch.Tensor: An 1d tensor in which each element is the entropy of the corresponding mention.
            The size of return tensor is identical to the size of input arg:`mention_logits`
    """
    new_mention_logits = mention_logits.unsqueeze(1)
    threshold_tensor = torch.full_like(new_mention_logits, threshold)
    mention_logits_with_threshold = torch.cat((threshold_tensor, new_mention_logits), dim=1)
    probs = F.softmax(mention_logits_with_threshold, dim=1)
    entropy = -torch.sum(probs * torch.log(probs), dim=1)
    return entropy


if __name__ == "__main__":
    t = torch.Tensor([1, 0, 2, -1])
    mention_detection_entropy(t)
