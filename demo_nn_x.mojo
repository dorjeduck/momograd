from momograd.x.engine import ValueX, ValueXList
from momograd.x.nn import NeuronX,LayerX,MLPX

fn main() raises:

    # Define input (xs) 
    var xs = VariadicList[ValueXList](
        ValueXList(2.0, 3.0, -1.0),
        ValueXList(3.0, -1.0, 0.5),
        ValueXList(3.0, -1.0, 0.5),
        ValueXList(1.0, 1.0, -1.0))

    var nin: Int = len(xs[0]) #number of input values

    #################
    # Single NeuronX #
    #################

    var nr:NeuronX = NeuronX(nin)

    print("Output single NeuronX:")
    print(nr(xs[0]).__str__())

    ################
    # Single LayerX #
    ################

    var ly:LayerX = LayerX(nin,5)
    
    print("\nOutput single LayerX:")
    print(ly(xs[0]).__str__())

    ############
    # Demo MLPX #
    ############

    print("\nDemo MLPX\n")

    # Define ground truth  

    var ys = ValueXList(1.0, -1.0, -1.0, 1.0)

    # Setup neural network input size and output layers structure.
   
    var nouts = VariadicList[Int](4, 4, 1)
    var model = MLPX(nin, nouts)

    # Set learning rate and number of training epochs.
    var learning_rate = 0.01
    var n_epochs = 100

    # Retrieve model parameters for training.
    var params = model.parameters()

    # training variables
    var ypred = ValueXList(len(xs))
    var loss:ValueX = ValueX(0)

    # Begin training process.
    print("Training:\n")
    for i in range(n_epochs):   
  
        # Forward pass: Compute predictions (ypred) for inputs (xs).
        for i in range(len(xs)):
            ypred[i] = model(xs[i])[0]

        # Loss calculation: Sum squared errors between predictions (ypred) and actuals (ys).
        loss = ValueX(0)
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