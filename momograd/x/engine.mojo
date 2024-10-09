from collections.list import List 
from math import log,exp
from time import now


from .util import ValueXList

# Define a value struct that can be passed through computational graph nodes
@register_passable("trivial")
struct ValueX(CollectionElement, Stringable):
    # UnsafePointers for the value's data and its gradient for backpropagation
    var data_ptr: UnsafePointer[Float64]
    var grad_ptr: UnsafePointer[Float64]
    var label: StringRef

    # Previous values in the computation graph, and a function pointer for the backward pass
    var _prev: ValueXList
    var _backward: fn (
        prev: ValueXList, grad_ptr: UnsafePointer[Float64], data_ptr: UnsafePointer[Float64]
    ) -> None

    # Operation identifier and a timestamp to manage computation order
    var _op: StringRef
    var _topo_stamp:UnsafePointer[Int]

    # A static method that does nothing, used as the default backward operation
    @staticmethod
    fn nothing_to_do(
        prev: ValueXList, grad_ptr: UnsafePointer[Float64], data_ptr: UnsafePointer[Float64]
    ) -> None:
        pass

    fn __init__(inout self, data: Float64, label: StringRef = ""):
        self.data_ptr = UnsafePointer[Float64].alloc(1)
        self.data_ptr.store(data)
        self.grad_ptr = UnsafePointer[Float64].alloc(1)
        self.grad_ptr.store(0.0)

        self._prev = ValueXList(0)
        self._backward = ValueX.nothing_to_do

        self.label = label
        self._op = " "

        # topo helper
        self._topo_stamp = UnsafePointer[Int]().alloc(1)
        self._topo_stamp[0] = 0

    fn __init__(
        inout self,
        data: Float64,
        prev: ValueXList,
        op: StringRef,
        label: StringRef = '',
    ):
        self.data_ptr = UnsafePointer[Float64].alloc(1)
        self.data_ptr.store(data)
        self.grad_ptr = UnsafePointer[Float64].alloc(1)
        self.grad_ptr.store(0.0)

        self.label = label

        # internal variables used for autograd graph construction
        self._backward = ValueX.nothing_to_do
        self._prev = prev
        self._op = op

        # topo helper
        self._topo_stamp = UnsafePointer[Int]().alloc(1)
        self._topo_stamp[0] = 0

    
    # Operator overloads for adding, multiplying etc, creating new Values in the graph

    @always_inline
    fn __add__(self, other: ValueX) -> ValueX:
        var prev = ValueXList(2)
        prev[0] = self
        prev[1] = other

        var out = ValueX(self.data_ptr[0] + other.data_ptr[0], prev, "+")

        fn _backward(
            prev: ValueXList, grad_ptr: UnsafePointer[Float64], data_ptr: UnsafePointer[Float64]
        ) -> None:
            prev[0].grad_ptr.store(prev[0].grad_ptr[0] + grad_ptr[0])
            prev[1].grad_ptr.store(prev[1].grad_ptr[0] + grad_ptr[0])

        out._backward = _backward

        return out

    @always_inline
    fn __mul__(self, other: ValueX) -> ValueX:
        var prev = ValueXList(2)
        prev[0] = self
        prev[1] = other

        var out = ValueX(self.data_ptr[0] * other.data_ptr[0], prev, "*")

        fn _backward(
            prev: ValueXList, grad_ptr: UnsafePointer[Float64], data_ptr: UnsafePointer[Float64]
        ) -> None:
            prev[0].grad_ptr.store(
                prev[0].grad_ptr[0] + prev[1].data_ptr[0] * grad_ptr[0]
            )
            prev[1].grad_ptr.store(
                prev[1].grad_ptr[0] + prev[0].data_ptr[0] * grad_ptr[0]
            )

        out._backward = _backward

        return out

    @always_inline
    fn __pow__(self, other: ValueX) -> ValueX:
        var prev = ValueXList(2)
        prev[0] = self
        prev[1] = other

        var out = ValueX(self.data_ptr[0] ** other.data_ptr[0], prev, "**")

        fn _backward(
            prev: ValueXList, grad_ptr: UnsafePointer[Float64], data_ptr: UnsafePointer[Float64]
        ) -> None:
            prev[0].grad_ptr.store(
                prev[0].grad_ptr[0]
                + grad_ptr[0]
                * (
                    prev[1].data_ptr[0]
                    * prev[0].data_ptr[0] ** (prev[1].data_ptr[0] - 1)
                )
            )
            prev[1].grad_ptr.store(
                prev[1].grad_ptr[0]
                + grad_ptr[0]
                * log(prev[0].data_ptr[0])
                * (prev[0].data_ptr[0] ** prev[1].data_ptr[0])
            )

        out._backward = _backward

        return out

    @always_inline
    fn tanh(self) -> ValueX:
        var prev = ValueXList(1)
        prev[0] = self
        
        var val:Float64 = (exp(2*self.data_ptr[0]) - 1)/(exp(2*self.data_ptr[0]) + 1)
        
        var out = ValueX(val, prev, "tanh")

        fn _backward(
            prev: ValueXList, grad_ptr: UnsafePointer[Float64], data_ptr: UnsafePointer[Float64]
        ) -> None:
            # tanh(x) d/dx = 1 - tanh(x)^2
            prev[0].grad_ptr.store(
                prev[0].grad_ptr[0] + (1-data_ptr[0]**2) * grad_ptr[0]
            )

        out._backward = _backward

        return out

    @always_inline
    fn relu(self) -> ValueX:

        var prev = ValueXList(1)
        prev[0] = self

        var val: Float64 = self.data_ptr[0]
        if val <= 0:
            val = 0
        var out = ValueX(val, prev, "ReLU")

        fn _backward(
            prev: ValueXList, grad_ptr: UnsafePointer[Float64], data_ptr: UnsafePointer[Float64]
        ) -> None:
            if data_ptr[0] > 0:
                prev[0].grad_ptr[0] += grad_ptr[0]

        out._backward = _backward

        return out

    @always_inline
    fn __add__(self, other: Float64) -> ValueX:
        return self + ValueX(other)

    @always_inline
    fn __iadd__(inout self,other:ValueX) -> None:
        # A new ValueX is created to maintain the integrity of the computational graph.
        # The label is transferred to the new ValueX.      
        var label = self.label
        self.label = ''
        self = self + other  # not sure if Mojo is happy with this ;-)
        self.label = label 
    
    @always_inline
    fn __isub__(inout self,other:ValueX) -> None:
        # A new ValueX is created to maintain the integrity of the computational graph.
        # The label is transferred to the new ValueX.
        var label = self.label
        self.label = ''
        self = self - other
        self.label = label
    
    @always_inline
    fn __neg__(self) -> ValueX:
        return self * -1

    @always_inline
    fn __sub__(self, other: Float64) -> ValueX:
        return self + (-other)

    @always_inline
    fn __sub__(self, other: ValueX) -> ValueX:
        return self + (-other)

    @always_inline
    fn __mul__(self, other: Float64) -> ValueX:
        return self * ValueX(other)

    @always_inline
    fn __truediv__(self, other: Float64) -> ValueX:
        return self * other**-1

    @always_inline
    fn __truediv__(self, other: ValueX) -> ValueX:
        return self * other**-1

    @always_inline
    fn __pow__(self, other: Float64) -> ValueX:
        return self.__pow__(ValueX(other))

    # --- reverse ...

    @always_inline
    fn __radd__(self, other: Float64) -> ValueX:
        return self + other

    @always_inline
    fn __rsub__(self, other: Float64) -> ValueX:
        return (-self) + other

    @always_inline
    fn __rmul__(self, other: Float64) -> ValueX:
        return self * other

    @always_inline
    fn __rtruediv__(self, other: Float64) -> ValueX:  # other / self
        return other * self**-1

    # Performs the backward pass, computing gradients in reverse topological order.
    fn backward(self) raises:

        # topological order all of the children in the graph
        var topo: List[ValueX] = List[ValueX]()
        ValueX._build_topo(self, topo,now())
        topo.reverse()
      
        topo[0].grad_ptr.store(1.0)
        ## calculating the gradients
        for i in range(len(topo)):
            if len(topo[i]._prev) > 0:
                topo[i]._backward(topo[i]._prev, topo[i].grad_ptr, topo[i].data_ptr)

    # Builds a topological order of the computation graph for the backward pass.
    @staticmethod
    fn _build_topo(value: ValueX, inout topo: List[ValueX],stamp:Int):
        if value._topo_stamp[0] == stamp:
            return
        value._topo_stamp[0] = stamp # mark value as visited for this topo run

        for i in range(len(value._prev)):
            ValueX._build_topo(value._prev[i], topo,stamp)

        topo.append(value)

    # Returns a string representation of the ValueX, including data and gradient.
    fn __str__(self) -> String:
        var out = "<data: "
            + str(self.data_ptr[0])
            + ", grad: "
            + str(self.grad_ptr[0])
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
