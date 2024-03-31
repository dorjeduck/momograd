
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

This repo contains the benchmark results from our local machine for reference. You might want to delete the `./benchmarks/results` folder before running your own benchmarks.
