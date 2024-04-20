import torch
import torch.nn as nn
import torch.nn.functional as F
import CRASP

class Transformer(nn.Module):
    def __init__(self, hidden_size):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 0  # Initially, there are no layers

        # Initialize lists to store self-attention and feed-forward layers
        self.self_attentions = nn.ModuleList([])
        self.feed_forwards = nn.ModuleList([])

        # Store layer normalization layers
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, elementwise_affine=False)

    def add_self_attention_layer(self):
        # Append a new self-attention layer
        self.self_attentions.append(nn.MultiheadAttention(self.hidden_size, num_heads = 1))
        self.num_layers += 1

    def add_feed_forward_layer(self):
        # Append a new feed-forward layer
        feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
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

# Create a Transformer object with hidden_size=512 and num_heads=8
transformer = Transformer(hidden_size=512)

# Add layers manually
transformer.add_self_attention_layer()
transformer.add_feed_forward_layer()
transformer.add_self_attention_layer()
transformer.add_feed_forward_layer()  # Add this line

# Example usage
input_data = torch.randn((10, 20, 512))  # Example input data
output = transformer(input_data)
print(output.shape)  # Should be (10, 20, 512)