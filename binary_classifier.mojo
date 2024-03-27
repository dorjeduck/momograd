from python import Python
from python.object import PythonObject
from time import now

from random import seed
from sys import argv
from collections.vector import InlinedFixedVector

from momograd.engine import Value, ValueList
from momograd.nn import Neuron, Layer, MLP
from momograd.util import append_to_file

def logging(txt: String, silent: Bool = False) -> None:
    if not silent:
        print(txt)


fn main() raises:
    seed(37)

    var N_EPOCHS = 50  # number of training steps
    var N_SAMPLES = 50  # number of training samples
    var SILENT = False

    var CSV_FILE_PATH: String = "benchmark_mojo.csv"
    var BENCHMARK_CSV = False

    var args = argv()

    #  ... to enable benchmark automation ;-)
    if len(args) > 1:
        N_EPOCHS = atol(args[1])
        if len(args) > 2:
            N_SAMPLES = atol(args[2])
            if len(args) > 3 and args[3] == "--silent":
                SILENT = True
                if len(args) > 4 and args[4] == "--csv":
                    BENCHMARK_CSV = True
                    if len(args) > 5:
                        CSV_FILE_PATH = args[5]

    var nin = 2
    var nouts = VariadicList[Int](16, 16, 1)

    var model = MLP(nin, nouts)
    var params = model.parameters()

    logging("Number of parameters: " + str(len(params)), SILENT)

    var skdata = Python.import_module("sklearn.datasets")
    var out = skdata.make_moons(N_SAMPLES)

    var npx = out[0]
    var npy = out[1]

    var input = InlinedFixedVector[ValueList](N_SAMPLES)
    var yb = InlinedFixedVector[Value](N_SAMPLES)

    for i in range(N_SAMPLES):
        input[i] = ValueList(npx[i][0].to_float64(), npx[i][1].to_float64())
       
        yb[i] = 2 * Value(npy[i].to_float64()) - 1

    var scores = ValueList(N_SAMPLES)

    var data_loss = Value(0)
    var reg_loss = Value(0)
    var accuracy: Float16 = 0.0
    var learning_rate: Float64 = 0.1

    var start_time = now()

    for epoch in range(N_EPOCHS):
        # Forward pass: Compute scores for inputs
        for i in range(N_SAMPLES):
            scores[i] = model(input[i])[0]

        # compute loss
        data_loss = Value(0)
        accuracy = 0.0

        # svm "max-margin" loss
        for i in range(N_SAMPLES):
            data_loss += (1 - scores[i] * yb[i]).relu()
            if (scores[i].data_ptr.load() > 0) == (yb[i].data_ptr.load() > 0):
                accuracy += 1

        accuracy /= N_SAMPLES
        data_loss = data_loss / N_SAMPLES

        reg_loss = Value(0)
        for i in range(len(params)):
            reg_loss += params[i].load() * params[i].load()

        var total_loss = 1e-4 * reg_loss + data_loss

        # Print loss for this epoch.
        logging(
            "Epoch "
            + str(epoch)
            + ": loss = "
            + str(total_loss.data_ptr.load())
            + " (accuracy:"
            + str(accuracy * 100)
            + "%)",
            SILENT,
        )

        # Zero gradients before backpropagation.
        for i in range(len(params)):
            params[i].load().grad_ptr.store(0.0)

        # Backward pass: Compute gradients.
        total_loss.backward()

        learning_rate = 1.0 - 0.9 * epoch / 100
        # Parameter update: Apply gradient descent.
        for i in range(len(params)):
            if params[i].load().grad_ptr.load() != 0.0:
                params[i].load().data_ptr.store(
                    params[i].load().data_ptr.load()
                    - learning_rate * params[i].load().grad_ptr.load()
                )

    var elapsed_time = now() - start_time

    if BENCHMARK_CSV:
        append_to_file(
            CSV_FILE_PATH,
            str(N_SAMPLES)
            + ","
            + N_EPOCHS
            + ","
            + elapsed_time / 1000000000
            + ","
            + accuracy,
            "n_samples,n_epochs,time,accuracy",
        )
