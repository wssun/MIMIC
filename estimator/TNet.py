import torch.nn as nn
import torch


def tile_and_concat(tensor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Merge 1D and 2D tensor (use to aggregate feature maps and representation
    and compute local mutual information estimation)

    Args:
        tensor (torch.Tensor): 2D tensor (feature maps)
        vector (torch.Tensor): 1D tensor representation

    Returns:
        torch.Tensor: Merged tensor (2D)
    """

    B, C, H, W = tensor.size()
    vector = vector.unsqueeze(2).unsqueeze(2)
    expanded_vector = vector.expand((B, vector.size(1), H, W))
    return torch.cat([tensor, expanded_vector], dim=1)


class TNet(nn.Module):
    """Global statistics network

    Args:
        feature_map_size (int): Size of input feature maps
        feature_map_channels (int): Number of channels in the input feature maps
        latent_dim (int): Dimension of the representationss
    """

    def __init__(
        self, feature_map_size: int, feature_map_channels: int, latent_dim: int
    ):

        super().__init__()
        # self.flatten = nn.Flatten(start_dim=1)
        # print((feature_map_size ** 2 * feature_map_channels) + latent_dim)
        # print(feature_map_size, latent_dim)
        # import sys
        # sys.exit(-1)
        self.dense1 = nn.Linear(
            in_features=(feature_map_size ** 2 * feature_map_channels) + latent_dim,
            out_features=512,
        )
        self.dense2 = nn.Linear(in_features=512, out_features=512)
        self.dense3 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()

    def forward(
        self, feature_map: torch.Tensor, representation: torch.Tensor
    ) -> torch.Tensor:
        # print(feature_map.shape, representation.shape)
        # feature_map = self.flatten(feature_map)
        # import sys
        # sys.exit(-1)
        x = torch.cat([feature_map, representation], dim=-1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        global_statistics = self.dense3(x)
        return global_statistics
