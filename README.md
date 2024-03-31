# momograd

![''](/imgs/momograd.jpg)

## A Learning Journey: Micrograd in Mojo 🔥

This project represents an implementation of Andrej Karpathy's  Python based [micrograd](https://github.com/karpathy/micrograd) library in the [Mojo](https://docs.modular.com/mojo) programming language.

`micrograd` is a tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API. For an in depth explanation of what it is about and it's implementation details, don't miss to watch Andrej's excellent YouTube video [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

 `momograd` aims to follow `micrograd's` clean implementation structure with no intention to go beyond its functionality, but to learn how things can be done in Mojo. Expect to encounter bugs and sharp edges here.

 `momograd.x` ventures further into exploring Mojo's unique capabilities, focusing on optimizations such as vectorization and parallelization. This extension serves as a playground for delving into more advanced Mojo-specific enhancements, pushing beyond the original implementation logic of `micrograd` to explore how performance and efficiency can be improved within the Mojo environment.

## momograd.engine

The `Value` struct of `momograd.engine` represents the basic building block for the computational graph.

``` python
from momograd.engine import Value

fn main() raises:

    var a = Value(3,"a")
    var b = Value(2,"b")
    
    var c = a + b
    c.label = 'c' # optional label 

    var d = c ** a
    d.label = 'd'

    # calulating the gradients of the computational graph
    d.backward()

    print(d) # <data: 125.0, grad: 1.0>

    print("\nComputational graph:")
    d.print_branch() 
```

See [demo_engine.mojo](https://github.com/dorjeduck/momograd/blob/main/demo_engine.mojo) for a more elaborate example.

## momograd.nn

Following the implementation of `micrograd`, `momograd.nn` contains two structs for building neuronal networks, `Neuron` and `Layer`, and an implementation of a Multi-Layer Perceptron, `MLP`.

See [demo_nn.mojo](https://github.com/dorjeduck/momograd/blob/main/demo_nn.mojo) for a basic example how to use these structs in Mojo.

## momograd.x

Playground for Mojo specific optimizations.

## Benchmarks

The `micrograd` github repository includes a full demo of training an 2-layer neural network (MLP) binary classifier
([demo.ipynb](https://github.com/karpathy/micrograd/blob/master/demo.ipynb)). In order to be able to run basic benchmark comparisons, we include the core of it in [binary_classifier.py](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.py), and reimplemented it in Mojo using `momograd`: [binary_classifier.mojo](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.mojo).
[binary_classifier_x.mojo](https://github.com/dorjeduck/momograd/blob/main/binary_classifier_x.mojo) takes advantage of our experimental momograd.x package. 

Please take the following benchmark results with a grain of salt. We basically ignored everything mentioned in the excellent blog post by Konstantinos Krommydas, [How to Be Confident in Your Performance Benchmarking](https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking). Here we just measured the time the training loops took for 100 epochs and various sample size inputs for [binary_classifier.py](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.py) and [binary_classifier.mojo](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.mojo), averaged over a couple of runs, and joyfully observed how fast Mojo actually is.

&nbsp;

<div align="center">

| samples| micrograd (sec) | momograd (sec) | momograd.x (sec) | Speedup micro/momo | Speedup micro/momo.x | Speedup momo/momo.x |
| --- | --- |---| --- | --- | ---| --- |
| 20 | 5.64 | 0.15 | 0.13 | 36.6x | 44.9x | 1.2x |
| 40 | 18.37 | 0.31 | 0.23 | 59.1x | 79.8x | 1.3x |
| 60 | 35.06 | 0.48 | 0.35 | 72.7x | 100.1x | 1.4x |
| 80 | 53.93 | 0.63 | 0.48 | 86.1x | 112.6x | 1.3x |
| 100 | 73.91 | 0.82 | 0.61 | 90.5x | 122.0x | 1.3x |
| 120 | 94.01 | 0.97 | 0.71 | 96.8x | 132.3x | 1.4x |
| 140 | 113.46 | 1.16 | 0.84 | 97.7x | 134.4x | 1.4x |
| 160 | 131.64 | 1.28 | 0.91 | 102.5x | 144.6x | 1.4x |
| 180 | 149.30 | 1.46 | 1.01 | 102.6x | 148.5x | 1.4x |
| 200 | 168.71 | 1.70 | 1.28 | 99.1x | 131.7x | 1.3x |

&nbsp;

![''](/imgs/chart_time_comparison.png)

&nbsp;

![''](/imgs/chart_speedup_comparison.png)

</div>

For instructions on running benchmarks, see [Benchmark Instructions](benchmarks/BENCHMARK_INSTRUCTIONS.md).

## License

MIT
