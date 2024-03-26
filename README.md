# momograd

![''](/imgs/momograd.jpg)

### A Learning Journey: Micrograd in Mojo 🔥

This project represents an implementation of Andrej Karpathy's  Python based [micrograd](https://github.com/karpathy/micrograd) library in the [Mojo](https://docs.modular.com/mojo) programming language.

`micrograd` is a tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API. For an in depth explanation of what it is about and it's implementation details, don't miss to watch Andrej's excellent YouTube video [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

 `momograd` aims to follow `micrograd's` clean implementation structure with no intention to go beyond its functionality, but to learn how things can be done in Mojo. Expect to encounter bugs and rough edges here.

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

### momograd.nn

Following the implementation of `micrograd`, `momograd.nn` contains two structs for building neuronal networks, `Neuron` and `Layer`, and an implementation of a Multi-Layer Perceptron, `MLP`.

See [demo_nn.mojo](https://github.com/dorjeduck/momograd/blob/main/demo_nn.mojo) for a basic example how to use these structs in Mojo.

### Benchmarks

The `micrograd` github repository includes a full demo of training an 2-layer neural network (MLP) binary classifier
([demo.ipynb](https://github.com/karpathy/micrograd/blob/master/demo.ipynb)). In order to be able to run basic benchmark comparisons, we include the core of it in [binary_classifier.py](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.py), and reimplemented it in Mojo using `momograd`: [binary_classifier.mojo](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.mojo).

Please take the following benchmark results with a grand of salt. We basically ignored everything mentioned in the excellent blog post by Konstantinos Krommydas, [How to Be Confident in Your Performance Benchmarking](https://www.modular.com/blog/how-to-be-confident-in-your-performance-benchmarking). Here we just measured the time the training loops took for various sample size inputs for [binary_classifier.py](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.py) and [binary_classifier.mojo](https://github.com/dorjeduck/momograd/blob/main/binary_classifier.mojo), averaged over a couple of runs, and joyfully observed how fast Mojo actually is.

&nbsp;

<div align="center">

| samples | micrograd (sec) | momograd (sec) | Speed Up |
|---------|---------------------------|-----------------------|----------|
| 20 | 5.64 | 0.17 | 33.1x |
| 40 | 18.37 | 0.33 | 55.3x |
| 60 | 35.06 | 0.52 | 67.7x |
| 80 | 53.93 | 0.61 | 88.1x |
| 100 | 73.91 | 0.80 | 92.8x |
| 120 | 94.01 | 0.94 | 100.3x |
| 140 | 113.46 | 1.11 | 102.6x |
| 160 | 131.64 | 1.26 | 104.7x |
| 180 | 149.30 | 1.42 | 105.4x |
| 200 | 168.71 | 1.70 | 99.3x |

&nbsp;

![''](/imgs/chart_time_comparison.png)

&nbsp;

![''](/imgs/chart_speedup_comparison.png)

</div>

We include the scripts for running these basic benchmarks in the `benchmarks` folder. On a unix like system you can run these in the terminal:

Besides Mojo, install the required Python libs:

```bash
pip install matplotlib, micrograd, numpy, pandas, scikit-learn
```

Execute the following commands to run the benchmarks, average and combine the benchmark results and create the charts and markdown table.

``` bash
cd ./benchmarks

# Usage: ./run_benchmarks.sh <epochs> 'sample_sizes' [rounds]
./run_benchmarks.sh 100 '20 50 100 150 200' 5

python combine_benchmark_results.py
python create_charts_and_md.py
```

This repo contains the benchmark results from my local machine for reference. You might want to delete the `./benchmarks/results` folder before running your own benchmarks.

### License

MIT
