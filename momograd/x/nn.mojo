from collections.list import List 
from random import seed, random_float64
from .engine import ValueX, ValueXList

from algorithm import parallelize


# Define the NeuronX structure with weights, bias, and activation function
@register_passable("trivial")
struct NeuronX:
    var w: ValueXList  # Weight values for the neuron
    var b: ValueX  # Bias value for the neuron
    var b_ptr: Pointer[ValueX]  # Pointer to the bias to facilitate updates
    var nin: Int  # Number of inputs to the neuron
    var nonlin: Bool  # Boolean flag to use a non-linear activation function

    # Initialize neuron with random weights and bias
    fn __init__(inout self, nin: Int, nonlin: Bool = True):
        self.w = ValueXList(nin)
        self.b = ValueX(random_float64(-1, 1))

        self.b_ptr = Pointer[ValueX].alloc(1)
        self.b_ptr.store(self.b)

        self.nin = nin
        self.nonlin = nonlin

        for i in range(nin):
            self.w[i] = ValueX(random_float64(-1, 1))

    # Define how a neuron processes input values
    fn __call__(self, input: ValueXList) -> ValueX:
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
    fn add_parameters(self, inout params: List[Pointer[ValueX]]) -> None:
        for i in range(self.nin):
            params.append(self.w.get_val_ptr(i))
        params.append(self.b_ptr)


# Define the LayerX structure containing multiple neurons
@register_passable("trivial")
struct LayerX:
    var neurons: Pointer[NeuronX]  # Dynamic array of neurons in the layer
    var nin: Int  # Number of inputs to the layer
    var nout: Int  # Number of outputs/neurons in the layer

    # Initialize the layer with specified number of neurons and optional non-linearity
    fn __init__(inout self, nin: Int, nout: Int, nonlin: Bool = True):
        self.nin = nin
        self.nout = nout

        self.neurons = Pointer[NeuronX].alloc(nout)

        for i in range(nout):
            self.neurons[i] = NeuronX(nin, nonlin)

    # Define how the layer processes input values
    fn __call__(self, input: ValueXList) -> ValueXList:
        var result = ValueXList(self.nout)

        # Pass input through each neuron and return the outputs
        @parameter
        fn _call_neurons(i: Int):
            result[i] = self.neurons[i](input)
        parallelize[_call_neurons](self.nout, self.nout)
       
        return result

    # Collecting layer parameters
    @always_inline
    fn add_parameters(self, inout params: List[Pointer[ValueX]]) -> None:
        for i in range(self.nout):
            self.neurons[i].add_parameters(params)


# Define of a MLPX (Multi-LayerX Perceptron) structure with multiple layers
@register_passable("trivial")
struct MLPX:
    var layers: Pointer[LayerX]  # Dynamic array of layers in the MLPX
    var nin: Int  # Number of inputs to the MLPX
    var num_layers: Int  # Total number of layers in the MLPX

    # Initialize the MLPX with specified layer configurations
    fn __init__(inout self, nin: Int, nouts: VariadicList[Int]):
        self.nin = nin
        self.num_layers = len(nouts)
        self.layers = Pointer[LayerX].alloc(self.num_layers)

        # Initialize each layer based on the configuration
        self.layers.store(0, LayerX(nin, nouts[0], True))

        for i in range(1, self.num_layers):
            self.layers.store(i, LayerX(nouts[i - 1], nouts[i], i < self.num_layers - 1))

    # Define how the MLPX processes input values through its layers
    fn __call__(self, input: ValueXList) -> ValueXList:
        var result = input

        for i in range(self.num_layers):
            result = self.layers[i](result)

        return result

    # Collects and returns all trainable parameters of the MLPX.
    fn parameters(self) -> List[Pointer[ValueX]]:
        var params = List[Pointer[ValueX]]()

        for i in range(self.num_layers):
            self.layers[i].add_parameters(params)

        return params
