# C-RASP

This is an implementation of the compilation from C-RASP programs into transformer encoders, as specified in [the paper](https://arxiv.org/abs/2404.04393). More details to come... :)

# TODO items
Implement Bools
- And
- OR
- Comparison...

Implement Counts
- Conditional
- Addition
- Subtraction
- Min
- Max
- Constants
- Design choices: Hide BOS. Also need another FFN to ensure the BOS count is always 0, otherwise it screws with the rest

Utilities
- When pretty-printing, mask out the counts that have been overwritten by LayerNorm during Comparison
- Make a more user-friendly interface for writing programs - a parser?
- Make it very streamlined to use as language acceptor/rejector
- Companion program to just simulate C-RASP programs

Language Modeling
- I said we could define simple language models using C-RASP, so we should test it out
- Presumably this can be used to edit existing language models. For instance, we can force it to never generate X if it has not seen at least 3 Y's. This can be done by adding fresh dimensions and inserting a C-RASP program which checks whether or not there have been 3 Y's or not. Then we can add a huge coefficient to this Boolean value and add it to the dimension which stores the probability of X being the next token. We'd also have to muck around with the final layer, but depending on how the language model is defined this is maybe ok.

More
- How much of this can be applied to S-RASP?
- Experiments -- initialize a C-RASP transformer and fine-tune on something