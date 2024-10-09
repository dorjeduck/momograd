from momograd.engine import Value, ValueList
from momograd.nn import Neuron,Layer,MLP

fn main() raises:

    # Define input (xs) 
    var xs = VariadicList[ValueList](
        ValueList(2.0, 3.0, -1.0),
        ValueList(3.0, -1.0, 0.5),
        ValueList(3.0, -1.0, 0.5),
        ValueList(1.0, 1.0, -1.0))

    var nin: Int = len(xs[0]) #number of input values

    #################
    # Single Neuron #
    #################

    var nr:Neuron = Neuron(nin)

    print("Output single Neuron:")
    print(nr(xs[0]).__str__())

    ################
    # Single Layer #
    ################

    var ly:Layer = Layer(nin,5)
    
    print("\nOutput single Layer:")
    print(ly(xs[0]).__str__())

    ############
    # Demo MLP #
    ############

    print("\nDemo MLP\n")

    # Define ground truth  

    var ys = ValueList(1.0, -1.0, -1.0, 1.0)

    # Setup neural network input size and output layers structure.
   
    var nouts = VariadicList[Int](4, 4, 1)
    var model = MLP(nin, nouts)

    # Set learning rate and number of training epochs.
    var learning_rate = 0.01
    var n_epochs = 100

    # Retrieve model parameters for training.
    var params = model.parameters()

    # training variables
    var ypred = ValueList(len(xs))
    var loss:Value = Value(0)

    # Begin training process.
    print("Training:\n")
    for i in range(n_epochs):   
  
        # Forward pass: Compute predictions (ypred) for inputs (xs).
        for i in range(len(xs)):
            ypred[i] = model(xs[i])[0]

        # Loss calculation: Sum squared errors between predictions (ypred) and actuals (ys).
        loss = Value(0)
        for i in range(len(xs)):
            loss += (ypred[i] - ys[i])**2

        # Print loss for this epoch.
        print("Epoch " + str(i) + ": loss =", loss.data_ptr.load())

        # Zero gradients before backpropagation.
        for i in range(len(params)):   
            params[i][0].grad_ptr.store(0.0)

        # Backward pass: Compute gradients.
        loss.backward()

        # Parameter update: Apply gradient descent.
        for i in range(len(params)):
            if params[i][0].grad_ptr.load() != 0.0:
                params[i][0].data_ptr.store(params[i][0].data_ptr.load() - learning_rate * params[i][0].grad_ptr.load())

        
    # Display prediction and corresponding ground truth values.
    print("\nPrediction:\n")

    for i in range(len(ypred)):
        print(str(ypred[i].data_ptr.load()) + " (ground truth: " + str(ys[i].data_ptr.load()) + ")")