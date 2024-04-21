import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, dims):
        super(Transformer, self).__init__()
        self.dims = dims
        self.num_layers = 0  # Initially, there are no layers

        # Initialize lists to store self-attention and feed-forward layers
        self.self_attentions = nn.ModuleList([])
        self.feed_forwards = nn.ModuleList([])
        self.layers = nn.ModuleList([])

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
        self.layers.append(self_attention)
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
        self.layers.append(self_attention)
        self.num_layers += 1

    def add_feed_forward_layer(self):
        # Append a new feed-forward layer
        feed_forward = nn.Sequential(
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims)
        )
        self.feed_forwards.append(feed_forward)
        self.layers.append(feed_forward)
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
        self.layers.append(feed_forward)
        self.num_layers += 1

    def forward(self, x):
        layer_output = x
        for i in range(self.num_layers):
            if self.layers[i].__class__.__name__ == "MultiheadAttention":
                # Self-Attention Layer
                mask = torch.triu(torch.ones((1, x.size(0), x.size(0))), diagonal=1).bool()
                layer_output, _ = self.layers[i](x, x, x, attn_mask=mask)
                # x = self.layer_norm(x + layer_output)
            else:
                # Feed-Forward Layer
                layer_output = self.layers[i](x)
                # x = self.layer_norm(x + layer_output)
        return layer_output

# Verified computation by hand on a simple uniform attention computation

# mydim = 2
# transformer = Transformer(dims=mydim)

# custom_query_weight = torch.zeros((mydim, mydim))
# custom_key_weight = torch.zeros((mydim, mydim))
# custom_value_weight = torch.eye(mydim)
# transformer.add_self_attention_layer_custom(custom_query_weight, custom_key_weight, custom_value_weight)

# for name, param in transformer.named_parameters():
#     # Check if is in_proj_weight, and if so print the 3 separate weight matrices
#     if "in_proj_weight" in name:
#         w_q, w_k, w_v = param.chunk(3)
#         print(f"Layer: {name}")
#         print(f"Query weights: {w_q}")
#         print(f"Key weights: {w_k}")
#         print(f"Value weights: {w_v}")
#     else:
#         print(f"Layer: {name}")
#         print(f"Weights: {param.data}")

# # Example usage
# input_data = torch.randint(0, 2, (4, mydim)).float()
# print(input_data)
# output = transformer(input_data)
# print(output)
