import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """Supervised/unsupervised normalized temperature-scaled cross entropy loss.

    Supports unsupervised loss of SimCLR (https://arxiv.org/abs/2002.05709) and
        supervised/live loss of Positional Contrastive Learning for Volumetric
        Medical Image Segmentation (https://arxiv.org/abs/2106.09157).

    Attributes:
        temperature (float, optional): Temperature used for scaling of the
            similarity values. Defaults to 0.07.
        threshold (float | None, optional): Positive pair thresholds for a
            supervised version of the loss. Can be set to None to force unsupervised
            contrastive loss (causing error when providing labels). Defaults to 0.1.
        cyclic_relative_labels (bool, optional): evaluate labels as cyclic (assuming
            interval [0, 1]). Defaults to False.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        threshold: float | None = 0.1,
        cyclic_relative_labels: bool = False,
    ):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.cyclic_relative_labels = cyclic_relative_labels
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        """Compute supervised loss from features and labels, or unsupervised loss from
            features.

        Expects a batch where every 2 features come from 2 different augmentations of
            the same image. This results in the many 2 * batch_size shapes, where
            batch_size is the original batch size of unqiue images, before making any
            copies for augmentation.

        Args:
            features (torch.Tensor): features, shape: [2 * batch_size, feature_size]
            labels (torch.Tensor | None, optional): labels for live contrastive pairing,
                shape: [2 * batch_size]. Can be set to None for unsupervised contrastive
                loss. Defaults to None.

        Returns:
            torch.Tensor: normalized temperature-scaled cross entropy loss
        """

        nr_features = features.shape[0]

        # mask to not make features positive pairs with themselves
        self_contrast_mask = 1.0 - torch.eye(nr_features, device=features.device)

        # make predetermined unsupervised or live label-based supervised mask to only
        # consider positive pairs (multiply negative pair values by zero), a symmetrical
        # square matrix of shape: [2 * batch_size, 2 * batch_size]

        # unsupervised contrastive loss (SimCLR)
        if labels is None:
            # make features positive pairs with their corresponding augmented version
            # results in matrix with pattern (identity with every 2x2 block flipped):
            # [[0, 1, 0, 0, 0, 0],
            #  [1, 0, 0, 0, 0, 0],
            #  [0, 0, 0, 1, 0, 0],
            #  [0, 0, 1, 0, 0, 0],
            #  [0, 0, 0, 0, 0, 1],
            #  [0, 0, 0, 0, 1, 0]]
            # Since the augmented versions of the same image are the only positive pairs
            # in SimCLR, and the order of augmented versions is always the same when
            # using the qcardia-data pipeline, this pattern is always the same and can
            # be hardcoded.
            pattern = torch.zeros(nr_features - 1, device=features.device)
            pattern[::2] = 1  # make [1, 0, 1, 0, ..., 1, 0, 1] pattern next to diagonal
            positive_pair_mask = torch.diag(pattern, 1) + torch.diag(pattern, -1)

        # supervised contrastive loss (Positional Contrastive Learning)
        else:
            labels = labels.unsqueeze(-1)
            deviation = torch.abs(
                labels.T.repeat(nr_features, 1) - labels.repeat(1, nr_features)
            )  # label deviation matrix for all possible feature pairs

            if self.cyclic_relative_labels:
                # consider the label a cyclic value where 0 and 1 are equivalent.
                # example: deviation = abs(0.95 - 0.02) = 0.93
                # becomes: deviation = 1 - abs(0.95 - 0.02) = 0.07
                deviation = torch.abs((deviation > 0.5).float() - deviation)

            # mask to only consider positive pairs (label deviation below threshold)
            positive_pair_mask = (
                deviation < self.threshold
            ).float() * self_contrast_mask

        # scaled cosine similarity matrix for all feature pairs
        scaled_similarities = (
            self.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0))
            / self.temperature
        )

        # calculate exponential of the scaled cosine similarities, masked to set
        # similarities of each feature with itself to zero.
        masked_exp_similarities = torch.exp(scaled_similarities) * self_contrast_mask

        # simplify `log(exp(scaled_similarities) / sum(masked_exp_similarities))`
        # to `scaled_similarities - log(sum(masked_exp_similarities))`, by using
        # `log(a / b) = log(a) - log(b)` and `log(exp(a)) = a`
        feature_pair_losses = scaled_similarities - torch.log(
            masked_exp_similarities.sum(1, keepdim=True)
        )  # matrix of (negative) losses for all feature pairs

        # mask feature pair loss to set negative pair losses to zero
        pos_feature_pair_losses = feature_pair_losses * positive_pair_mask

        # vector of mean loss for each feature (sum of all (positive) feature pair
        # losses divided by the number of positive pairs for each feature)
        feature_losses = pos_feature_pair_losses.sum(1) / positive_pair_mask.sum(1)

        # mean loss of batch, with changed sign to make it a minimization problem
        return -feature_losses.mean()
