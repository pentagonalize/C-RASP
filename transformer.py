import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, dims, use_layer_norm=True):
        super(Transformer, self).__init__()
        self.dims = dims
        self.use_layer_norm = use_layer_norm
        self.num_layers = 0  # Initially, there are no layers

        # Initialize lists to store self-attention and feed-forward layers
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
        # transpose the value weight matrix to match the expected shape
        self_attention.out_proj.weight = nn.Parameter(custom_value_weight)
        # Set bias to zero for both in_proj and out_proj
        self_attention.in_proj_bias = nn.Parameter(torch.zeros(3 * self.dims))
        self_attention.out_proj.bias = nn.Parameter(torch.zeros(self.dims))

        # Add to list
        self.layers.append(self_attention)
        self.num_layers += 1

    def add_feed_forward_layer(self):
        # Append a new feed-forward layer
        feed_forward = nn.Sequential(
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims)
        )
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
        linear_1 = nn.Linear(self.dims, hidden_dims, bias=False)
        linear_1.weight.data = custom_feed_forward_1
        linear_2 = nn.Linear(hidden_dims, self.dims, bias=False)
        linear_2.weight.data = custom_feed_forward_2

        # Append a new feed-forward layer
        feed_forward = nn.Sequential(
            linear_1,
            nn.ReLU(),
            linear_2
        )
        self.layers.append(feed_forward)
        self.num_layers += 1

    def forward(self, x):
        layer_output = x
        for i in range(self.num_layers):
            prev_output = layer_output
            if self.layers[i].__class__.__name__ == "MultiheadAttention":
                # Self-Attention Layer
                mask = torch.triu(torch.ones((1, prev_output.size(0), prev_output.size(0))), diagonal=1).bool()
                layer_output, _ = self.layers[i](prev_output, prev_output, prev_output, attn_mask=mask)
                # Residual connection
                layer_output = layer_output + prev_output
                # Layer normalization
                if self.use_layer_norm:
                    layer_output = self.layer_norm(layer_output)
            else:
                # Feed-Forward Layer
                layer_output = self.layers[i](layer_output)
                # Residual connection
                layer_output = layer_output + prev_output
                # Layer normalization
                if self.use_layer_norm:
                    layer_output = self.layer_norm(layer_output)
        return layer_output

# Verified attention by hand on a simple uniform attention computation

# mydim = 4
# transformer = Transformer(dims=mydim)

# custom_query_weight = torch.zeros((mydim, mydim))
# custom_key_weight = torch.zeros((mydim, mydim))
# custom_value_weight = torch.zeros((mydim, mydim))
# custom_value_weight[3, 0] = 1
# custom_value_weight[3, 3] = 1

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

# # set the last column to all 0
# input_data[:, 3] = 0

# print(input_data)
# output = transformer(input_data)
# print(output)
