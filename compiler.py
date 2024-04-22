import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        word_embedding[word] = torch.tensor([[-1, 1] if j == i else [1, -1] for j in range(len(alphabet))] + [[-1, 1]]).flatten()
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

        # Add a constant
        self.Program.add_CONSTANT("TRUE")

        # Initialize an empty Transformer with dimensions double the alphabet size +1 for <|BOS|>
        # This is because the word embeddings are doubled up
        # Plus 2 more for the constant
        self.Transformer = transformer.Transformer(2*len(alphabet)+2)
        self.dims = 2*len(alphabet)+2

    def make_room(self):
        # We need to update the Transformer to reflect the new operation
        # First, expand the weight matrices of the Transformer to accomodate the new operation
        # Every attention matrix must get expanded by 2 in each direction
        for i in range(len(self.Transformer.layers)):
            layer = self.Transformer.layers[i]
            if layer.__class__.__name__ == "MultiheadAttention":
                # Here, we need to divide layer.in_proj_weight into 3 parts for key, query, and value
                key, query, value = layer.in_proj_weight.chunk(3)
                # Expand the each matrix by 2 in each direction and reset it
                expanded_key = expand_matrix(key, 2, 2)
                expanded_query = expand_matrix(query, 2, 2)
                expanded_value = expand_matrix(value, 2, 2)

                # The output projection matrix gets expanded by 2
                expanded_out_proj_weight = expand_matrix(layer.out_proj.weight.data, 2, 2)

                # Create a new MultiheadAttention layer with the updated dimensions
                new_layer = nn.MultiheadAttention(self.dims+2, layer.num_heads)
                new_layer.in_proj_weight = nn.Parameter(torch.cat([expanded_key, expanded_query, expanded_value], dim=0))
                new_layer.out_proj.weight.data = expanded_out_proj_weight

                # Replace the existing layer with the new layer
                self.Transformer.layers[i] = new_layer
            else:
                # Here, we need to expand the weight matrices of the two linear layers
                # The input of the first linear layer gets expanded by 2
                layer[0].weight.data = expand_matrix(layer[0].weight.data, 0, 2)
                layer[0].in_features = layer[0].weight.shape[1]
                # The output of the second linear layer gets expanded by 2
                layer[2].weight.data = expand_matrix(layer[2].weight.data, 2, 0)
                layer[2].out_features = layer[2].weight.shape[0]


        # The word embedding gets expanded by 2
        # That is two new rows of zeros are added to each word embedding
        for word in self.word_embedding:
            self.word_embedding[word] = torch.cat([self.word_embedding[word], torch.zeros(2)])

        # The Transformer dimensions get updated
        self.Transformer.dims += 2
        self.dims += 2

        # Update the LayerNorm
        self.Transformer.layer_norm = nn.LayerNorm(normalized_shape=self.dims, elementwise_affine=False)

        # # print the new dimensions of each layer for bug checking
        # print("checking")
        # for layer in self.Transformer.layers:
        #     if layer.__class__.__name__ == "MultiheadAttention":
        #         print(layer.in_proj_weight.shape)
        #         print(layer.out_proj.weight.shape)
        #     else:
        #         print(layer[0].weight.shape)
        #         print(layer[2].weight.shape)

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

    def add_COUNTING(self, operation_name, name):
        # Add a Counting operation to the CRASP program
        self.Program.add_COUNTING(operation_name, name)

        # Make room in the Transformer for the new operation
        self.make_room()

        # Now get the index of the operation in the CRASP program
        # This corresponds to the dimension in which the operation is stored in the Transformer
        # That is, 2*operation_index and 2*operation_index+1 store the Boolean values of the operation
        operation_index = self.Program.get_index(operation_name)

        # now define the query, key, and value matrices
        # the query and key will be all zeros
        custom_query_weight = torch.zeros((self.dims, self.dims))
        custom_key_weight = torch.zeros((self.dims, self.dims))

        # initialize the value matrix as all zeros
        custom_value_weight = torch.zeros((self.dims, self.dims))

        # set the correct routing values
        custom_value_weight[self.dims-2, self.dims-2] = 1
        custom_value_weight[self.dims-2, 2*operation_index] = 1

        custom_value_weight[self.dims-1, self.dims-1] = 1
        custom_value_weight[self.dims-1, 2*operation_index+1] = 1

        # Add the Counting operation to the Transformer
        self.Transformer.add_self_attention_layer_custom(custom_query_weight.float(), custom_key_weight.float(), custom_value_weight.float())

        # Construct the Feed-Forward Layer that fixes the count values
        # Locate the TRUE dimension
        true_index = self.Program.get_index("TRUE")

        # the first linear layer will be from self.dims to 2
        # basically the count is stored as -2X+1 in the negative dim
        # We retrieve X by multiplying by -1/2 and adding 1/2 (must be nonnegative so preserved by ReLU)
        custom_feed_forward_1 = torch.zeros((2, self.dims))
        custom_feed_forward_1[0, self.dims-2] = -0.5
        custom_feed_forward_1[0, 2*true_index] = -0.5
        custom_feed_forward_1[1, 2*true_index] = -1

        # The second linear layer will be from 2 to self.dims
        # Basically we add X and subtract 1 to -2x+1 to get -X as desired
        # Symmetric for the other dimension
        custom_feed_forward_2 = torch.zeros((self.dims, 2))
        custom_feed_forward_2[self.dims-2, 0] = 1
        custom_feed_forward_2[self.dims-2, 1] = -1
        custom_feed_forward_2[self.dims-1, 0] = -1
        custom_feed_forward_2[self.dims-1, 1] = 1

        # Add the Feed-Forward Layer to the Transformer
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

        if pretty_print:
            result = output_tensor
            # Delete every odd column
            result = result[:, ::2]
            # Make tensor a list of lists
            result = result.tolist()

            # First, note that LayerNorm applied a scaling factor to every column.
            # We need to undo this scaling factor
            # The magnitude of the TRUE column is the scaling factor
            # Some jank choices made here with negatives
            # Since the TRUE value is -1, we invert in various places to make it work out
            TRUE_index = self.Program.get_index("TRUE")
            for i in range(len(result)):
                scale = -result[i][TRUE_index]
                for j in range(len(result[i])):
                    result[i][j] = result[i][j]/scale

            for i in range(len(self.Program.operations)):
                # check if the ith operation is a BOOL operation
                if self.Program.operations[i].__class__.__bases__[0].__name__ == "BOOL":
                    for col in result:
                        if bool(math.isclose(col[i], 1, rel_tol=1e-5)):
                            col[i] = "T"
                        else:
                            col[i] = "F"
                else:
                    # scale by position to retrieve actual count value
                    for j in range(len(result)):
                        # Round also to nearest int
                        result[j][i] = round(result[j][i]*-(j+1))

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
                        # Add a vertical line after the first column
                        aligned_result += ' | ' if j == 0 else '  '
                # Newline after each row
                aligned_result += '\n'
                if i == 0:
                    # Horizontal line after the first row
                    aligned_result += '-' * (sum(column_widths) + 3 * len(column_widths) - 1) + '\n'

            return aligned_result
        else:
            return output_tensor

# Test the CRASP_to_Transformer class

alphabet = ['a', 'b']
model = CRASP_to_Transformer(alphabet)
model.add_NOT('Q_a', "P1")
model.add_COUNTING('Q_a', "P2")
model.add_COUNTING('P1', "P3")

print(model.word_embedding)
print(model.Program)
print(model.Transformer)

# Test the forward method
output = model(['b', 'a', 'a', 'b', 'a', 'b'], pretty_print=True)
print(output)

