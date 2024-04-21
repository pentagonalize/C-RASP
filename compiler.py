import torch
import torch.nn as nn
import torch.nn.functional as F
from CRASP import CRASP  # Import the CRASP class from the CRASP module
import transformer

# Some utility functions

def expand_matrix(matrix, x_dims, y_dims):
    # Expand a matrix to a higher dimension
    # For an mxn matrix, expand to a (m+x_dims)x(n+y_dims) matrix
    # The extra dimensions will be filled with zeros
    expanded_matrix = torch.zeros(matrix.shape[0]+x_dims, matrix.shape[1]+y_dims)
    expanded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return expanded_matrix

def word_embedding(alphabet):
    # For our purposes, a dictionary will suffice
    # The keys are the words in the vocabulary
    # The values are the one-hot encodings of the words with one modification:
    # Instead of just 1 and 0, we double it up and have [-1,1] represent 1 and [1,-1] represent 0
    word_embedding = {}
    for i, word in enumerate(alphabet):
        word_embedding[word] = torch.tensor([[-1, 1] if j == i else [1, -1] for j in range(len(alphabet))]).flatten()
    return word_embedding

# Define a CRASP to Transformer class

class CRASP_to_Transformer(nn.Module):
    def __init__(self, alphabet):
        super().__init__()
        self.alphabet = alphabet

        # Initilaize the word embedding matrix
        self.word_embedding = word_embedding(alphabet)

        # Initialize an empty CRASP program
        self.Program = CRASP(alphabet)

        # Initialize an empty Transformer with dimensions double the alphabet size +1 for <|BOS|>
        # This is because the word embeddings are doubled up
        self.Transformer = transformer.Transformer(2*len(alphabet)+2)

    def forward(self, input):
        # The input is a list of words
        # First convert the words to their one-hot encodings, to get a tensor of shape (len(input), 2*len(alphabet))
        # Then, pass this tensor through the Transformer

        # Convert the input to a tensor
        input_tensor = torch.stack([self.word_embedding[word] for word in input])

        # Pass the input tensor through the Transformer
        return self.Transformer(input_tensor)


# Test the CRASP_to_Transformer class

alphabet = ['a', 'b', 'c', 'd', 'e']
model = CRASP_to_Transformer(alphabet)
print(model.word_embedding)
print(model.Program)
print(model.Transformer)

# Test the forward method
output = model(['a', 'b', 'c'])
print(output)