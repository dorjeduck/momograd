from collections.list import List 
from math import log
from time import now


from .util import ValueList

# Define a value struct that can be passed through computational graph nodes
@register_passable("trivial")
struct Value(CollectionElement, Stringable):
    # Pointers for the value's data and its gradient for backpropagation
    var data_ptr: Pointer[Float64]
    var grad_ptr: Pointer[Float64]
    var label: StringRef

    # Previous values in the computation graph, and a function pointer for the backward pass
    var _prev: ValueList
    var _backward: fn (
        prev: ValueList, grad_ptr: Pointer[Float64], data_ptr: Pointer[Float64]
    ) -> None

    # Operation identifier and a timestamp to manage computation order
    var _op: StringRef
    var _topo_stamp:Pointer[Int]

    # A static method that does nothing, used as the default backward operation
    @staticmethod
    fn nothing_to_do(
        prev: ValueList, grad_ptr: Pointer[Float64], data_ptr: Pointer[Float64]
    ) -> None:
        pass

    fn __init__(inout self, data: Float64, label: StringRef = ""):
        self.data_ptr = Pointer[Float64].alloc(1)
        self.data_ptr.store(data)
        self.grad_ptr = Pointer[Float64].alloc(1)
        self.grad_ptr.store(0.0)

        self._prev = ValueList(0)
        self._backward = Value.nothing_to_do

        self.label = label
        self._op = " "

        # topo helper
        self._topo_stamp = Pointer[Int]().alloc(1)
        self._topo_stamp.store(0)

    fn __init__(
        inout self,
        data: Float64,
        _prev: ValueList,
        op: StringRef,
        label: StringRef = '',
    ):
        self.data_ptr = Pointer[Float64].alloc(1)
        self.data_ptr.store(data)
        self.grad_ptr = Pointer[Float64].alloc(1)
        self.grad_ptr.store(0.0)

        self.label = label

        # internal variables used for autograd graph construction
        self._backward = Value.nothing_to_do
        self._prev = _prev
        self._op = op

        # topo helper
        self._topo_stamp = Pointer[Int]().alloc(1)
        self._topo_stamp.store(0)

    
    # Operator overloads for adding, multiplying etc, creating new Values in the graph

    @always_inline
    fn __add__(self, other: Value) -> Value:
        var prev = ValueList(2)
        prev[0] = self
        prev[1] = other

        var out = Value(self.data_ptr.load() + other.data_ptr.load(), prev, "+")

        fn _backward(
            prev: ValueList, grad_ptr: Pointer[Float64], data_ptr: Pointer[Float64]
        ) -> None:
            prev[0].grad_ptr.store(prev[0].grad_ptr.load() + grad_ptr.load())
            prev[1].grad_ptr.store(prev[1].grad_ptr.load() + grad_ptr.load())

        out._backward = _backward

        return out

    @always_inline
    fn __mul__(self, other: Value) -> Value:
        var prev = ValueList(2)
        prev[0] = self
        prev[1] = other

        var out = Value(self.data_ptr.load() * other.data_ptr.load(), prev, "*")

        fn _backward(
            prev: ValueList, grad_ptr: Pointer[Float64], data_ptr: Pointer[Float64]
        ) -> None:
            prev[0].grad_ptr.store(
                prev[0].grad_ptr.load() + prev[1].data_ptr.load() * grad_ptr.load()
            )
            prev[1].grad_ptr.store(
                prev[1].grad_ptr.load() + prev[0].data_ptr.load() * grad_ptr.load()
            )

        out._backward = _backward

        return out

    @always_inline
    fn __pow__(self, other: Value) -> Value:
        var prev = ValueList(2)
        prev[0] = self
        prev[1] = other

        var out = Value(self.data_ptr.load() ** other.data_ptr.load(), prev, "**")

        fn _backward(
            prev: ValueList, grad_ptr: Pointer[Float64], data_ptr: Pointer[Float64]
        ) -> None:
            prev[0].grad_ptr.store(
                prev[0].grad_ptr.load()
                + grad_ptr.load()
                * (
                    prev[1].data_ptr.load()
                    * prev[0].data_ptr.load() ** (prev[1].data_ptr.load() - 1)
                )
            )
            prev[1].grad_ptr.store(
                prev[1].grad_ptr.load()
                + grad_ptr.load()
                * log(prev[0].data_ptr.load())
                * (prev[0].data_ptr.load() ** prev[1].data_ptr.load())
            )

        out._backward = _backward

        return out

    @always_inline
    fn tanh(self) -> Value:
        var prev = ValueList(1)
        prev[0] = self
        
        var val:Float64 = (math.exp(2*self.data_ptr.load()) - 1)/(math.exp(2*self.data_ptr.load()) + 1)
        
        var out = Value(val, prev, "tanh")

        fn _backward(
            prev: ValueList, grad_ptr: Pointer[Float64], data_ptr: Pointer[Float64]
        ) -> None:
            # tanh(x) d/dx = 1 - tanh(x)^2
            prev[0].grad_ptr.store(
                prev[0].grad_ptr.load() + (1-data_ptr.load()**2) * grad_ptr.load()
            )

        out._backward = _backward

        return out

    @always_inline
    fn relu(self) -> Value:

        var prev = ValueList(1)
        prev[0] = self

        var val: Float64 = self.data_ptr.load()
        if val <= 0:
            val = 0
        var out = Value(val, prev, "ReLU")

        fn _backward(
            prev: ValueList, grad_ptr: Pointer[Float64], data_ptr: Pointer[Float64]
        ) -> None:
            if data_ptr.load() > 0:
                prev[0].grad_ptr.store(prev[0].grad_ptr.load() + grad_ptr.load())

        out._backward = _backward

        return out

    @always_inline
    fn __add__(self, other: Float64) -> Value:
        return self + Value(other)

    @always_inline
    fn __iadd__(inout self,other:Value) -> None:
        # A new Value is created to maintain the integrity of the computational graph.
        # The label is transferred to the new Value.      
        var label = self.label
        self.label = ''
        self = self + other  # not sure if Mojo is happy with this ;-)
        self.label = label 
    
    @always_inline
    fn __isub__(inout self,other:Value) -> None:
        # A new Value is created to maintain the integrity of the computational graph.
        # The label is transferred to the new Value.
        var label = self.label
        self.label = ''
        self = self - other
        self.label = label
    
    @always_inline
    fn __neg__(self) -> Value:
        return self * -1

    @always_inline
    fn __sub__(self, other: Float64) -> Value:
        return self + (-other)

    @always_inline
    fn __sub__(self, other: Value) -> Value:
        return self + (-other)

    @always_inline
    fn __mul__(self, other: Float64) -> Value:
        return self * Value(other)

    @always_inline
    fn __truediv__(self, other: Float64) -> Value:
        return self * other**-1

    @always_inline
    fn __truediv__(self, other: Value) -> Value:
        return self * other**-1

    @always_inline
    fn __pow__(self, other: Float64) -> Value:
        return self.__pow__(Value(other))

    # --- reverse ...

    @always_inline
    fn __radd__(self, other: Float64) -> Value:
        return self + other

    @always_inline
    fn __rsub__(self, other: Float64) -> Value:
        return (-self) + other

    @always_inline
    fn __rmul__(self, other: Float64) -> Value:
        return self * other

    @always_inline
    fn __rtruediv__(self, other: Float64) -> Value:  # other / self
        return other * self**-1

    # Performs the backward pass, computing gradients in reverse topological order.
    fn backward(self) raises:

        # topological order all of the children in the graph
        var topo: List[Value] = List[Value]()
        Value._build_topo(self, topo,now())
        topo.reverse()
      
        topo[0].grad_ptr.store(1.0)
        ## calculating the gradients
        for i in range(len(topo)):
            if len(topo[i]._prev) > 0:
                topo[i]._backward(topo[i]._prev, topo[i].grad_ptr, topo[i].data_ptr)

    # Builds a topological order of the computation graph for the backward pass.
    @staticmethod
    fn _build_topo(value: Value, inout topo: List[Value],stamp:Int):
        if value._topo_stamp.load() == stamp:
            return
        value._topo_stamp.store(stamp) # mark value as visited for this topo run

        for i in range(len(value._prev)):
            Value._build_topo(value._prev[i], topo,stamp)

        topo.append(value)

    # Returns a string representation of the Value, including data and gradient.
    fn __str__(self) -> String:
        var out = "<data: "
            + str(self.data_ptr.load())
            + ", grad: "
            + str(self.grad_ptr.load())
            + ">"
        if len(self.label)>0:
            out += " (var " + str(self.label) + ") "
        return out
    
    # Recursively prints the branches of the computation graph.
    fn print_branch(self, depth: Int = 0):
        var ind: String = ""
        for i in range(depth):
            ind += "  "
        print(ind + self.__str__())
        if len(self._prev) > 0:
            print(ind + "  -------- (" + str(self._op) + ") --------")
            for i in range(len(self._prev)):
                self._prev[i].print_branch(depth + 1)
