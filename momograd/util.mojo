from collections.vector import InlinedFixedVector
from collections.list import List 

from .engine import Value


@register_passable("trivial")
struct ValueList(Sized, Stringable):
    var _values: LegacyPointer[Value]
    var _len: Int

    fn __init__(inout self, length: Int = 0):
        self._len = length
        if length > 0:
            self._values = LegacyPointer[Value].alloc(length)
            memset_zero(self._values, length)
        else:
            self._values = LegacyPointer[Value]()

    fn __init__(inout self, *vv: Float64):
        self._len = len(vv)
        if self._len > 0:
            self._values = LegacyPointer[Value].alloc(self._len)
            for i in range(self._len):
                self._values[i] = Value(vv[i])
        else:
            self._values = LegacyPointer[Value]()

    fn __init__(inout self, vv: VariadicList[Float64]):
        self._len = len(vv)
        if self._len > 0:
            self._values = LegacyPointer[Value].alloc(self._len)
            for i in range(self._len):
                self._values[i] = Value(vv[i])
        else:
            self._values = LegacyPointer[Value]()

    fn __init__(inout self, vv: List[Float64]):
        self._len = len(vv)
        if self._len > 0:
            self._values = LegacyPointer[Value].alloc(self._len)
            for i in range(self._len):
                self._values[i] = Value(vv[i])
        else:
            self._values = LegacyPointer[Value]()

    fn __len__(self) -> Int:
        return self._len

    fn __getitem__(self, idx: Int) -> Value:
        return self._values[idx]

    fn __setitem__(self, idx: Int, value: Value):
        self._values[idx] = value

    fn __str__(self) -> String:
        if self._len == 0:
            return "empty value list"
        var result = self._values[0].__str__()
        for i in range(1, self._len):
            result += "\n" + self._values[i].__str__()
        return result

    fn get_val_ptr(self, idx: Int) -> LegacyPointer[Value]:
        return self._values + idx


fn append_to_file(
    file_name: String, content: String, first_line_for_empty_file: String = ""
) raises:
    var f: FileHandle
    var prev: String = ""

    try:
        f = open(file_name, "r")
        prev = f.read()
        f.close()
    except:
        prev = first_line_for_empty_file

    f = open(file_name, "w")
    f.write(prev + "\n" + content)
    f.close()
