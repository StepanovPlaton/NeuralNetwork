import tensor.tensor as T

if (T.MODE == T.PLATFORM.OPENCL):
    T.init()


class Layer:
    inputFeatures: int
    outputFeatures: int
    weights: T.Matrix
    bias: T.Matrix  # T.Vector
    activation: T.FUNCTION

    internal: T.Matrix  # T.Vector
    outputs: T.Matrix  # T.Vector

    def __init__(self, inputFeatures: int, outputFeatures: int, activation: T.FUNCTION):
        self.inputFeatures = inputFeatures
        self.outputFeatures = outputFeatures
        self.weights = T.Matrix([outputFeatures, inputFeatures], 0, 1)*0.1
        self.bias = T.Matrix([outputFeatures, 1], 0)
        self.activation = activation

        self.internal = T.Matrix([outputFeatures, 1], 0)
        self.outputs = T.Matrix([outputFeatures, 1], 0)


class NN:
    layers: list[Layer]

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, inputs: T.Matrix) -> T.Matrix:
        for i, layer in enumerate(self.layers):
            layer.internal = (
                layer.weights @
                (inputs if i == 0 else self.layers[i-1].outputs)
            ) + layer.bias
            layer.outputs = layer.internal(layer.activation)
        return self.layers[len(self.layers)-1].outputs

    def learn(self, inputs: T.Matrix, target: T.Matrix):
        self.forward(inputs)

        lossVector = self.layers[len(self.layers) -
                                 1].outputs - target
        # print("loss", lossVector(T.FUNCTION.MSE))
        dAnl = lossVector(T.FUNCTION.MSE, True)
        for i in range(len(self.layers)-1, -1, -1):
            dZl = dAnl * \
                self.layers[i].internal(self.layers[i].activation, True)
            dWl = dZl @ (inputs if i ==
                         0 else self.layers[i-1].outputs).t()
            dbl = dZl
            # dbl = dZl.sum(axis=1).reshape(dZl.shape[0], 1)
            dAnl = self.layers[i].weights.t() @ dZl
            self.layers[i].weights.t()
            self.layers[i].weights += (dWl * -0.3)
            self.layers[i].bias += (dbl * -0.3)


nn = NN([Layer(2, 3, T.FUNCTION.SIGMOID), Layer(3, 1, T.FUNCTION.LINEAR)])

print("Обучение...")
for epoch in range(1000):
    total_loss = 0
    for i in range(0, 2):
        for j in range(0, 2):
            input = T.Matrix([2, 1], [i, j])
            output = T.Matrix([1, 1], [i ^ j])
            nn.learn(input, output)

    if epoch % 100 == 0:
        print(f"Эпоха {epoch}")
        for i in range(0, 2):
            for j in range(0, 2):
                input = T.Matrix([2, 1], [i, j])
                predicted = nn.forward(input)
                print(
                    f"{i} XOR {j} = {i ^ j}, NN: ", predicted)
        print()

print("Финальные результаты:")
for i in range(0, 2):
    for j in range(0, 2):
        input = T.Matrix([2, 1], [i, j])
        predicted = nn.forward(input)
        print(
            f"{i} XOR {j} = {i ^ j}, NN: ", predicted)
print()
