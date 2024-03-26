from random import seed, random_float64
from .engine import Value, ValueList

# Define the Neuron structure with weights, bias, and activation function
@register_passable("trivial")
struct Neuron:
    var w: ValueList  # Weight values for the neuron
    var b: Value  # Bias value for the neuron
    var b_ptr: Pointer[Value]  # Pointer to the bias to facilitate updates
    var nin: Int  # Number of inputs to the neuron
    var nonlin: Bool  # Boolean flag to use a non-linear activation function

    # Initialize neuron with random weights and bias
    fn __init__(inout self, nin: Int, nonlin: Bool = True):
        self.w = ValueList(nin)
        self.b = Value(random_float64(-1, 1))

        self.b_ptr = Pointer[Value].alloc(1)
        self.b_ptr.store(self.b)

        self.nin = nin
        self.nonlin = nonlin

        for i in range(nin):
            self.w[i] = Value(random_float64(-1, 1))

    # Define how a neuron processes input values
    fn __call__(self, input: ValueList) -> Value:
        var result = self.b  # Start with the bias

        # Compute the weighted sum of inputs
        for i in range(self.nin):
            result = result + self.w[i] * input[i]

        # Apply the non-linear activation function if specified
        if self.nonlin:
            return result.relu()
        else:
            return result

    # Add neuron parameters to a dynamic vector for optimization
    @always_inline
    fn add_parameters(self, inout params: DynamicVector[Pointer[Value]]) -> None:
        for i in range(self.nin):
            params.push_back(self.w.get_val_ptr(i))
        params.push_back(self.b_ptr)


# Define the Layer structure containing multiple neurons
@register_passable("trivial")
struct Layer:
    var neurons: Pointer[Neuron]  # Dynamic array of neurons in the layer
    var nin: Int  # Number of inputs to the layer
    var nout: Int  # Number of outputs/neurons in the layer

    # Initialize the layer with specified number of neurons and optional non-linearity
    fn __init__(inout self, nin: Int, nout: Int, nonlin: Bool = True):
        self.nin = nin
        self.nout = nout

        self.neurons = Pointer[Neuron].alloc(nout)

        for i in range(nout):
            self.neurons[i] = Neuron(nin, nonlin)

    # Define how the layer processes input values
    fn __call__(self, input: ValueList) -> ValueList:
        var result = ValueList(self.nout)

        # Pass input through each neuron and return the outputs
        for i in range(self.nout):
            result[i] = self.neurons[i](input)

        return result

    # Collecting layer parameters
    @always_inline
    fn add_parameters(self, inout params: DynamicVector[Pointer[Value]]) -> None:
        for i in range(self.nout):
            self.neurons[i].add_parameters(params)


# Define of a MLP (Multi-Layer Perceptron) structure with multiple layers
@register_passable("trivial")
struct MLP:
    var layers: Pointer[Layer]  # Dynamic array of layers in the MLP
    var nin: Int  # Number of inputs to the MLP
    var num_layers: Int  # Total number of layers in the MLP

    # Initialize the MLP with specified layer configurations
    fn __init__(inout self, nin: Int, nouts: VariadicList[Int]):
        self.nin = nin
        self.num_layers = len(nouts)
        self.layers = Pointer[Layer].alloc(self.num_layers)

        # Initialize each layer based on the configuration
        self.layers.store(0, Layer(nin, nouts[0], True))

        for i in range(1, self.num_layers):
            self.layers.store(i, Layer(nouts[i - 1], nouts[i], i < self.num_layers - 1))

    # Define how the MLP processes input values through its layers
    fn __call__(self, input: ValueList) -> ValueList:
        var result = input

        for i in range(self.num_layers):
            result = self.layers[i](result)

        return result

    # Collects and returns all trainable parameters of the MLP.
    fn parameters(self) -> DynamicVector[Pointer[Value]]:
        var params = DynamicVector[Pointer[Value]]()

        for i in range(self.num_layers):
            self.layers[i].add_parameters(params)

        return params
