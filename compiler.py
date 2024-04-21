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

    # If <|BOS|> is not in the alphabet, add it
    if "<|BOS|>" not in alphabet:
        alphabet.append("<|BOS|>")
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
        self.Transformer = transformer.Transformer(2*len(alphabet))
        self.dims = 2*len(alphabet)

    def make_room(self):
        # We need to update the Transformer to reflect the new operation
        # First, expand the weight matrices of the Transformer to accomodate the new operation
        # Every attention matrix must get expanded by 2 in each direction
        for layer in self.Transformer.layers:
            if layer.__class__.__name__ == "MultiheadAttention":
                # Here, we need to divide layer.in_proj_weight into 3 parts for key, query, and value
                key, query, value = layer.in_proj_weight.chunk(3)
                # Expand the each matrix by 2 in each direction and reset it
                layer.in_proj_weight = torch.cat([expand_matrix(key, 2, 2), expand_matrix(query, 2, 2), expand_matrix(value, 2, 2)], dim=0)

                # The output projection matrix gets expanded by 2
                layer.out_proj.weight.data = expand_matrix(layer.out_proj.weight.data, 2, 2)
            else:
                # Here, we need to expand the weight matrices of the two linear layers
                # The input of the first linear layer gets expanded by 2
                layer[0].weight.data = expand_matrix(layer[0].weight.data, 0, 2)
                # The output of the second linear layer gets expanded by 2
                layer[2].weight.data = expand_matrix(layer[2].weight.data, 2, 0)

        # The word embedding gets expanded by 2
        # That is two new rows of zeros are added to each word embedding
        for word in self.word_embedding:
            self.word_embedding[word] = torch.cat([self.word_embedding[word], torch.zeros(2)])

        # The Transformer dimensions get updated
        self.Transformer.dims += 2
        self.dims += 2

        # Update the LayerNorm
        self.Transformer.layer_norm = nn.LayerNorm(normalized_shape=self.dims, elementwise_affine=False)

    def add_NOT(self, operation_name, name):
        # Add a NOT operation to the CRASP program that negates the output of the operation with the given name
        self.Program.add_NOT(operation_name, name)

        # Make room in the Transformer for the new operation
        self.make_room()

        # Now get the index of the operation in the CRASP program
        # This corresponds to the dimension in which the operation is stored in the Transformer
        # That is, 2*operation_index and 2*operation_index+1 store the Boolean values of the operation
        operation_index = self.Program.get_index(operation_name)

        # Set the first linear layer so as to compute NOT
        custom_feed_forward_1 = torch.zeros((2, self.dims))
        custom_feed_forward_1[0, 2*operation_index] = 1
        custom_feed_forward_1[1, 2*operation_index+1] = 1

        # Set the second linear layer so as to compute NOT
        custom_feed_forward_2 = torch.zeros((self.dims, 2))
        custom_feed_forward_2[self.dims-2, 0] = -1
        custom_feed_forward_2[self.dims-1, 0] = 1
        custom_feed_forward_2[self.dims-1, 1] = -1
        custom_feed_forward_2[self.dims-2, 1] = 1

        # Add the NOT operation to the Transformer
        self.Transformer.add_feed_forward_layer_custom(custom_feed_forward_1.float(), custom_feed_forward_2.float())

    def forward(self, input, pretty_print=False):
        # The input is a list of words
        # First convert the words to their one-hot encodings, to get a tensor of shape (len(input), 2*len(alphabet))
        # Then, pass this tensor through the Transformer

        # First append the <|BOS|> token to the input
        input = ["<|BOS|>"] + input

        # Convert the input to a tensor
        input_tensor = torch.stack([self.word_embedding[word] for word in input]).float()

        # Pass the input tensor through the Transformer
        output_tensor = self.Transformer(input_tensor)

        # Get the value in the last two dimensions of the last position and convert to a Boolean value [-1,1] -> True, [1,-1] -> False
        result = output_tensor[-1, -2:].flatten()
        result = bool(result[0] == -1)

        if pretty_print:
            result = output_tensor
            # Delete every odd column
            result = result[:, ::2]
            # Make tensor a list of lists
            result = result.tolist()
            for i in range(len(self.Program.operations)):
                # check if the ith operation is a BOOL operation
                if self.Program.operations[i].__class__.__bases__[0].__name__ == "BOOL":
                    for col in result:
                        if col[i] == -1:
                            col[i] = "T"
                        else:
                            col[i] = "F"
                else:
                    # multiply the
                    for j in range(len(result)):
                        result[j][i] = result[j][i]*j

            # reformat so the program trace is comprehensible
            result = [[result[j][i] for j in range(len(result))] for i in range(len(result[0]))]
            # append the operation name to the beginning of each row
            for i in range(len(result)):
                result[i] = [self.Program.operations[i].verbose_str()] + result[i]
            # add the input to the beginning of the list
            result = [[""] + input] + result

            # reformat
            column_widths = [max(len(str(item)) for item in col) for col in zip(*result)]

            # Generate the aligned result as a string
            aligned_result = ''
            for i, row in enumerate(result):
                for j, (col, width) in enumerate(zip(row, column_widths)):
                    aligned_result += f"{col:>{width}}"
                    if j < len(row) - 1:
                        aligned_result += ' | ' if j == 0 else '  '  # Add a vertical line after the first column
                aligned_result += '\n'  # Newline after each row
                if i == 0:
                    aligned_result += '-' * (sum(column_widths) + 3 * len(column_widths) - 1) + '\n'  # Horizontal line after the first row

            return aligned_result
        else:
            return output_tensor

# Test the CRASP_to_Transformer class

alphabet = ['a', 'b', 'c', 'd', 'e']
model = CRASP_to_Transformer(alphabet)
model.add_NOT('Q_a', "P1")
model.add_NOT('Q_c', "P2")
model.add_NOT('Q_d', "P3")

print(model.word_embedding)
print(model.Program)
print(model.Transformer)

# Test the forward method
output = model(['a','b','c'], pretty_print=True)
print(output)