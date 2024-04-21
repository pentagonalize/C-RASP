import torch
import torch.nn as nn
import torch.nn.functional as F
import CRASP

class Transformer(nn.Module):
    def __init__(self, dims):
        super(Transformer, self).__init__()
        self.dims = dims
        self.num_layers = 0  # Initially, there are no layers

        # Initialize lists to store self-attention and feed-forward layers
        self.self_attentions = nn.ModuleList([])
        self.feed_forwards = nn.ModuleList([])

        # Store layer normalization layers
        self.layer_norm = nn.LayerNorm(normalized_shape=dims, elementwise_affine=False)

    def add_self_attention_layer(self):
        # Append a new self-attention layer
        self_attention = nn.MultiheadAttention(self.dims, num_heads = 1)
        custom_query_weight = torch.randint(0, 2, (self.dims, self.dims)).float()
        custom_key_weight = torch.randint(0, 2, (self.dims, self.dims)).float()
        custom_value_weight = torch.randint(0, 2, (self.dims, self.dims)).float()

        # Assign the custom weight matrices to the MultiheadAttention module
        self_attention.in_proj_weight = nn.Parameter(torch.cat([custom_query_weight, custom_key_weight, custom_value_weight], dim=0))
        self_attention.out_proj.weight = nn.Parameter(custom_value_weight)  # Since value and output projections are the same

        self.self_attentions.append(self_attention)
        self.num_layers += 1

    def add_self_attention_layer_custom(self, custom_query_weight, custom_key_weight, custom_value_weight):
        # Append a new self-attention layer
        self_attention = nn.MultiheadAttention(self.dims, num_heads = 1)

        # The module requires the weight matrices to be of shape (dims, dims), I think...
        assert custom_query_weight.shape == (self.dims, self.dims), "Query weight matrix has incorrect shape"
        assert custom_key_weight.shape == (self.dims, self.dims), "Key weight matrix has incorrect shape"
        assert custom_value_weight.shape == (self.dims, self.dims), "Value weight matrix has incorrect shape"

        # Assign the custom weight matrices to the MultiheadAttention module
        self_attention.in_proj_weight = nn.Parameter(torch.cat([custom_query_weight, custom_key_weight, custom_value_weight], dim=0))
        self_attention.out_proj.weight = nn.Parameter(custom_value_weight)  # Since value and output projections are the same

        self.self_attentions.append(self_attention)
        self.num_layers += 1

    def add_feed_forward_layer(self):
        # Append a new feed-forward layer
        feed_forward = nn.Sequential(
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims)
        )
        self.feed_forwards.append(feed_forward)
        self.num_layers += 1

    def add_feed_forward_layer_custom(self, custom_feed_forward_1, custom_feed_forward_2):
        # Append a new feed-forward layer

        # custom_feed_forward_1 and custom_feed_forward_2 are matrices
        assert custom_feed_forward_1.shape[1] == self.dims, "Feed-forward weight matrix 1 has incorrect shape"
        assert custom_feed_forward_1.shape[0] == custom_feed_forward_2.shape[1], "Feed-forward weight matrices have incompatible shapes"
        assert custom_feed_forward_2.shape[0] == self.dims, "Feed-forward weight matrix 2 has incorrect shape"

        hidden_dims = custom_feed_forward_1.shape[0]

        # make nn.Linear objects using the custom weights
        linear_1 = nn.Linear(self.dims, hidden_dims)
        linear_1.weight.data = custom_feed_forward_1
        linear_2 = nn.Linear(hidden_dims, self.dims)
        linear_2.weight.data = custom_feed_forward_2

        # Append a new feed-forward layer
        feed_forward = nn.Sequential(
            linear_1,
            nn.ReLU(),
            linear_2
        )
        self.feed_forwards.append(feed_forward)
        self.num_layers += 1

    def forward(self, x):
        num_layers = len(self.self_attentions)
        for i in range(num_layers):
            # Self-Attention Layer
            self_attention_output, _ = self.self_attentions[i](x, x, x)
            x = self.layer_norm(x + self_attention_output)

            # Feed-Forward Layer
            feed_forward_output = self.feed_forwards[i](x)
            x = self.layer_norm(x + feed_forward_output)
        return x

# Create a Transformer object with dims=512 and num_heads=8
mydim = 10
transformer = Transformer(dims=mydim)


# Add layers manually
transformer.add_self_attention_layer()
transformer.add_feed_forward_layer()

# Add a self-attention layer with custom weights
custom_query_weight = torch.randn((mydim, mydim))
custom_key_weight = torch.randn((mydim, mydim))
custom_value_weight = torch.randn((mydim, mydim))
transformer.add_self_attention_layer_custom(custom_query_weight, custom_key_weight, custom_value_weight)

# Add a feed-forward layer with custom weights
custom_feed_forward_1 = torch.randn((20, mydim))
custom_feed_forward_2 = torch.randn((mydim, 20))
transformer.add_feed_forward_layer_custom(custom_feed_forward_1, custom_feed_forward_2)

# for name, param in transformer.named_parameters():
#     print(f"Layer: {name}")
#     print(f"Weights: {param.data}")

# Example usage
input_data = torch.randn((10, 20, mydim))  # Example input data
output = transformer(input_data)
print(output.shape)