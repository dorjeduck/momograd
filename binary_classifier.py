import csv
import argparse
import random
import numpy as np

import time

from sklearn.datasets import make_moons, make_blobs

# pip install micrograd ...

from micrograd.engine import Value
from micrograd.nn import MLP

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments for N_EPOCHS, N_SAMPLES, and SILENT with their default values
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs (default: 50)')
parser.add_argument('--samples', type=int, default=100,
                    help='Number of training samples (default: 100)')
parser.add_argument('--silent', action='store_true',
                    help='Run in silent mode (default: False)')
parser.add_argument('--csv', action='store_true',
                    help='Write result to benchmark file (default: False)')
parser.add_argument('--csv_file_path', type=str, default="benchmark_py.csv",
                    help='Optional, file path for the csv (default: "benchmark_py.csv")')

# Parse the command line arguments
args = parser.parse_args()

# Assign the parsed arguments to variables
N_EPOCHS = args.epochs
N_SAMPLES = args.samples
SILENT = args.silent

BENCHMARK_CSV = args.csv
CSV_FILE_PATH = args.csv_file_path

np.random.seed(1337)
random.seed(1337)

X, y = make_moons(n_samples=N_SAMPLES)
y = y*2 - 1  # make y be -1 or 1

# initialize a model
model = MLP(2, [16, 16, 1])  # 2-layer neural network
if not SILENT:
    print(model)
    print("number of parameters", len(model.parameters()))

# loss function


def loss(batch_size=None):

    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0)
                for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


total_loss, acc = loss()

if not SILENT:
    # Print detailed information about the parameters
    print(
        f"Running Configuration:\n- Number of Training Epochs (N_EPOCHS): {N_EPOCHS}")
    print(f"- Number of Samples (N_SAMPLES): {N_SAMPLES}")

# Start measuring time at the beginning of the optimization loop
start_time = time.time()

# optimization
for k in range(N_EPOCHS):

    # forward
    total_loss, acc = loss()

    if not SILENT:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

    # backward
    model.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

# Calculate elapsed time after the loop completes
elapsed_time = time.time() - start_time

# Optionally, print the elapsed time
if not SILENT:
    print("Training completed. Total elapsed time: {:.2f} seconds".format(elapsed_time))

if BENCHMARK_CSV:
    # Open the file to append the benchmarking results

    with open(CSV_FILE_PATH, 'a', newline='') as file:
        writer = csv.writer(file)

        # Check if the file is empty to add the header
        file.seek(0, 2)  # Move to the end of the file to check its size
        if file.tell() == 0:  # File is empty, so we write the' header
            writer.writerow(['n_samples', 'n_epochs', 'time', 'accuracy'])

        # Append the benchmarking results
        writer.writerow([N_SAMPLES, N_EPOCHS, elapsed_time, acc])
