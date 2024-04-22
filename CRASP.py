# Defining C-RASP programs...

class BOOL():
    # Boolean Valued operations for CRASP
    # There are 5 types: Initial, Not, And, Comparison, and Constant

    def __init__(self, name):
        self.name = name

class COUNT():
    # Count Valued operations for CRASP
    # There are 5 types: Counting, Conditional, Addition, Subtraction, Min/Max, and Constant

    def __init__(self, name):
        self.name = name

# Defining the 5 types of BOOL operations

class INIT(BOOL):
    # Initial operation for CRASP
    # Takes in a symbol

    def __init__(self, symbol, name):
        super().__init__(name)
        self.symbol = symbol

    def __str__(self):
        return f"Q_{self.symbol}"

    def verbose_str(self):
        return f"Q_{self.symbol}"


class NOT(BOOL):
    # Not operation for CRASP
    # Takes in another BOOL operation

    def __init__(self, operation, name):
        assert isinstance(operation, BOOL), "operation must be an instance of BOOL"
        super().__init__(name)
        self.operation = operation

    def __str__(self):
        return f"~{self.operation.name}"

    def verbose_str(self):
        return f"~({self.operation.verbose_str()})"

class AND(BOOL):
    # And operation for CRASP
    # Takes in two BOOL operations

    def __init__(self, bool1, bool2, name):
        assert isinstance(bool1, BOOL), "first bool must be an instance of BOOL"
        assert isinstance(bool2, BOOL), "second bool must be an instance of BOOL"
        super().__init__(name)
        self.bool1 = bool1
        self.bool2 = bool2

    def __str__(self):
        return f"{self.bool1.name} & {self.bool2.name}"

    def verbose_str(self):
        return f"({self.bool1.verbose_str()} & {self.bool2.verbose_str()})"

class OR(BOOL):
    # Or operation for CRASP
    # Takes in two BOOL operations

    def __init__(self, bool1, bool2, name):
        assert isinstance(bool1, BOOL), "first bool must be an instance of BOOL"
        assert isinstance(bool2, BOOL), "second bool must be an instance of BOOL"
        super().__init__(name)
        self.bool1 = bool1
        self.bool2 = bool2

    def __str__(self):
        return f"{self.bool1.name} | {self.bool2.name}"

    def verbose_str(self):
        return f"({self.bool1.verbose_str()} | {self.bool2.verbose_str()})"

class COMPARE(BOOL):
    # Comparison operation for CRASP
    # Takes in two COUNT operations

    def __init__(self, count1, count2, name):
        assert isinstance(count1, COUNT), "first count must be an instance of COUNT"
        assert isinstance(count1, COUNT), "second count must be an instance of COUNT"
        super().__init__(name)
        self.count1 = count1
        self.count2 = count2

    def __str__(self):
        return f"{self.count1.name} < {self.count2.name}"

    def verbose_str(self):
        return f"({self.count1.verbose_str()} < {self.count2.verbose_str()})"

class CONSTANT(BOOL):
    # Constant operation for CRASP
    # Always returns True

    def __init__(self, name):
        super().__init__(name)
        pass

    def __str__(self):
        return "True"

    def verbose_str(self):
        return "True"

# Defining the 5 types of COUNT operations

class COUNTING(COUNT):
    # Counting Operation
    # Takes in a BOOL operation to count

    def __init__(self, operation, name):
        assert isinstance(operation, BOOL), "operation must be an instance of BOOL"
        super().__init__(name)
        self.operation = operation

    def __str__(self):
        return f"# {self.operation.name}"

    def verbose_str(self):
        return f"# {self.operation.verbose_str()}"

class CONDITIONAL(COUNT):
    # Conditional Operation
    # Takes in a BOOL operation and two COUNT operations
    # Uses C syntax for conditional

    def __init__(self, operation, count1, count2, name):
        assert isinstance(operation, BOOL), "operation must be an instance of BOOL"
        assert isinstance(count1, COUNT), "first count must be an instance of COUNT"
        assert isinstance(count2, COUNT), "second count must be an instance of COUNT"
        super().__init__(name)
        self.operation = operation
        self.count1 = count1
        self.count2 = count2

    def __str__(self):
        return f"{self.operation.name} ? {self.count1.name} : {self.count2.name}"

    def verbose_str(self):
        return f"({self.operation.verbose_str()} ? {self.count1.verbose_str()} : {self.count2.verbose_str()})"

class ADDITION(COUNT):
    # Addition Operation
    # Takes in two COUNT operations

    def __init__(self, count1, count2, name):
        assert isinstance(count1, COUNT), "first count must be an instance of COUNT"
        assert isinstance(count2, COUNT), "second count must be an instance of COUNT"
        super().__init__(name)
        self.count1 = count1
        self.count2 = count2

    def __str__(self):
        return f"{self.count1.name} + {self.count2.name}"

    def verbose_str(self):
        return f"({self.count1.verbose_str()} + {self.count2.verbose_str()})"

class SUBTRACTION(COUNT):
    # Subtraction Operation
    # Takes in two COUNT operations

    def __init__(self, count1, count2, name):
        assert isinstance(count1, COUNT), "first count must be an instance of COUNT"
        assert isinstance(count2, COUNT), "second count must be an instance of COUNT"
        super().__init__(name)
        self.count1 = count1
        self.count2 = count2

    def __str__(self):
        return f"{self.count1.name} - {self.count2.name}"

    def verbose_str(self):
        return f"({self.count1.verbose_str()} - {self.count2.verbose_str()})"

class MIN(COUNT):
    # Min Operation
    # Takes in two COUNT operations

    def __init__(self, count1, count2, name):
        assert isinstance(count1, COUNT), "first count must be an instance of COUNT"
        assert isinstance(count2, COUNT), "second count must be an instance of COUNT"
        super().__init__(name)
        self.count1 = count1
        self.count2 = count2

    def __str__(self):
        return f"min({self.count1.name}, {self.count2.name})"

    def verbose_str(self):
        return f"min({self.count1.verbose_str()}, {self.count2.verbose_str()})"

class MAX(COUNT):
    # Max Operation
    # Takes in two COUNT operations

    def __init__(self, count1, count2, name):
        assert isinstance(count1, COUNT), "first count must be an instance of COUNT"
        assert isinstance(count2, COUNT), "second count must be an instance of COUNT"
        super().__init__(name)
        self.count1 = count1
        self.count2 = count2

    def __str__(self):
        return f"max({self.count1.name}, {self.count2.name})"

    def verbose_str(self):
        return f"max({self.count1.verbose_str()}, {self.count2.verbose_str()})"

class CONST(COUNT):
    # Constant Operation
    # Returns the specified integer

    def __init__(self, value, name):
        assert isinstance(value, int), "value must be an integer"
        super().__init__(name)
        self.value = value

    def __str__(self):
        return str(self.value)

    def verbose_str(self):
        return str(self.value)

class CRASP():
    # Class for CRASP program class
    # A CRASP program is a list of straighline operations, each of which can reference previous operations
    # The program is executed by iterating through the operations in order

    def __init__(self, alphabet):
        # We maintain that every operation has a unique name
        self.operations = []
        self.alphabet = alphabet

        # Add initial operations for each symbol in the alphabet
        for symbol in alphabet:
            self.operations.append(INIT(symbol, f"Q_{symbol}"))

        # add <|BOS|> if not in alphabet
        if "<|BOS|>" not in alphabet:
            self.operations.append(INIT("<|BOS|>", "Q_<|BOS|>"))

    def get_index(self, operation_name):
        # Get the index of the operation with the given name in the operations list
        return [operation.name for operation in self.operations].index(operation_name)

    def add_CONSTANT(self, name):
        # Add a CONSTANT operation
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        self.operations.append(CONSTANT(name))

    def add_NOT(self, operation_name, name):
        # Add a NOT operation

        # First check that the name is not already in use
        assert name not in [operation.name for operation in self.operations], "name must be unique"

        # Next check if the operation_name is a valid operation
        assert operation_name in [operation.name for operation in self.operations], "operation_name must be a valid operation name"

        # select the last operation with the given name
        operation = [operation for operation in self.operations if operation.name == operation_name][-1]

        # Add the NOT operation
        self.operations.append(NOT(operation, name))

    def add_AND(self, bool1_name, bool2_name, name):
        # Add an AND operation

        # First check that the name is not already in use
        assert name not in [operation.name for operation in self.operations], "name must be unique"

        print(bool1_name, bool2_name)
        # Next check if the bool1_name and bool2_name are valid operations
        assert bool1_name in [operation.name for operation in self.operations], "first bool name must be a valid operation name"
        assert bool2_name in [operation.name for operation in self.operations], "second bool name must be a valid operation name"

        # select the last operation with the given names
        bool1 = [operation for operation in self.operations if operation.name == bool1_name][-1]
        bool2 = [operation for operation in self.operations if operation.name == bool2_name][-1]

        # Add the AND operation
        self.operations.append(AND(bool1, bool2, name))

    def add_OR(self, bool1_name, bool2_name, name):
        # Add an OR operation

        # First check that the name is not already in use
        assert name not in [operation.name for operation in self.operations], "name must be unique"

        # Next check if the bool1_name and bool2_name are valid operations
        assert bool1_name in [operation.name for operation in self.operations], "first bool name must be a valid operation name"
        assert bool2_name in [operation.name for operation in self.operations], "second bool name must be a valid operation name"

        # select the last operation with the given names
        bool1 = [operation for operation in self.operations if operation.name == bool1_name][-1]
        bool2 = [operation for operation in self.operations if operation.name == bool2_name][-1]

        # Add the OR operation
        self.operations.append(OR(bool1, bool2, name))

    def add_COMPARE(self, count1_name, count2_name, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        assert count1_name in [operation.name for operation in self.operations], "first count name must be a valid operation name"
        assert count2_name in [operation.name for operation in self.operations], "second count name must be a valid operation name"

        # Check that the counts have not been overwritten - that is if there exists another comparison operation before this one
        # Get the minimum index of the two counts
        count1_index = self.get_index(count1_name)
        count2_index = self.get_index(count2_name)
        min_index = min(count1_index, count2_index)

        # Check if there is a comparison operation between min_index and the current index
        if any([isinstance(operation, COMPARE) for operation in self.operations[min_index:len(self.operations)]]):
            raise Exception("cannot compare counts that have been overwritten, need to recompute them (pls blame Ashish Vaswani, not me...)")

        count1 = [operation for operation in self.operations if operation.name == count1_name][-1]
        count2 = [operation for operation in self.operations if operation.name == count2_name][-1]

        self.operations.append(COMPARE(count1, count2, name))

    def add_COUNTING(self, operation_name, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        assert operation_name in [operation.name for operation in self.operations], "operation_name must be a valid operation name"

        operation = [operation for operation in self.operations if operation.name == operation_name][-1]

        self.operations.append(COUNTING(operation, name))

    def add_CONDITIONAL(self, operation_name, count1_name, count2_name, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        assert operation_name in [operation.name for operation in self.operations], "operation_name must be a valid operation name"
        assert count1_name in [operation.name for operation in self.operations], "first count name must be a valid operation name"
        assert count2_name in [operation.name for operation in self.operations], "second count name must be a valid operation name"

        operation = [operation for operation in self.operations if operation.name == operation_name][-1]
        count1 = [operation for operation in self.operations if operation.name == count1_name][-1]
        count2 = [operation for operation in self.operations if operation.name == count2_name][-1]

        self.operations.append(CONDITIONAL(operation, count1, count2, name))

    def add_ADDITION(self, count1_name, count2_name, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        assert count1_name in [operation.name for operation in self.operations], "first count name must be a valid operation name"
        assert count2_name in [operation.name for operation in self.operations], "second count name must be a valid operation name"

        count1 = [operation for operation in self.operations if operation.name == count1_name][-1]
        count2 = [operation for operation in self.operations if operation.name == count2_name][-1]

        self.operations.append(ADDITION(count1, count2, name))

    def add_SUBTRACTION(self, count1_name, count2_name, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        assert count1_name in [operation.name for operation in self.operations], "first count name must be a valid operation name"
        assert count2_name in [operation.name for operation in self.operations], "second count name must be a valid operation name"

        count1 = [operation for operation in self.operations if operation.name == count1_name][-1]
        count2 = [operation for operation in self.operations if operation.name == count2_name][-1]

        self.operations.append(SUBTRACTION(count1, count2, name))

    def add_MIN(self, count1_name, count2_name, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        assert count1_name in [operation.name for operation in self.operations], "first count name must be a valid operation name"
        assert count2_name in [operation.name for operation in self.operations], "second count name must be a valid operation name"

        count1 = [operation for operation in self.operations if operation.name == count1_name][-1]
        count2 = [operation for operation in self.operations if operation.name == count2_name][-1]

        self.operations.append(MIN(count1, count2, name))

    def add_MAX(self, count1_name, count2_name, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        assert count1_name in [operation.name for operation in self.operations], "first count name must be a valid operation name"
        assert count2_name in [operation.name for operation in self.operations], "second count name must be a valid operation name"

        count1 = [operation for operation in self.operations if operation.name == count1_name][-1]
        count2 = [operation for operation in self.operations if operation.name == count2_name][-1]

        self.operations.append(MAX(count1, count2, name))

    # # Removing CONST for now, layernorm might screw it up
    def add_CONST(self, value, name):
        assert name not in [operation.name for operation in self.operations], "name must be unique"
        self.operations.append(CONST(value, name))

    def __str__(self):
        return "\n".join([operation.name + " := " + str(operation) for operation in self.operations])

# Testing
# Create a CRASP program for the parity problem

# # Define the alphabet
# alphabet = ['0', '1']

# # Create a CRASP program
# example_program = CRASP(alphabet)

# # Add the NOT operation
# example_program.add_NOT("Q_0", "P")

# # Add the AND operation
# example_program.add_AND("Q_0", "P", "P2")

# # Add a COUNTING operation
# example_program.add_COUNTING("P2", "C")

# # Add an ADDITION operation
# example_program.add_ADDITION("C", "C", "C2")

# # Add a CONDITIONAL operation
# example_program.add_CONDITIONAL("P2", "C", "C2", "C3")

# # Add a MIN operation
# example_program.add_MIN("C", "C2", "C4")

# # Add a MAX operation
# example_program.add_MAX("C3", "C4", "C5")

# # Add a CONST operation
# example_program.add_CONST(1, "C6")

# # Print the program
# print(example_program)

