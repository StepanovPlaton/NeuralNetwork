#include "./math/math.hpp"

#include <chrono>
#include <thread>

using namespace GPU;

class Layer {
protected:
  int outputFeatures;
  Vector bias;
  Activation activation;
  float alpha;

public:
  Layer(int outputFeatures, Activation activation, Vector bias,
        float alpha = 0.0f)
      : outputFeatures(outputFeatures), bias(bias), activation(activation),
        alpha(alpha) {}

  int getOuputFeatures() const { return outputFeatures; }
  Activation getActivation() const { return activation; }
  float getAlpha() const { return alpha; }

  const Vector &getBias() const { return bias; }
};

class ConnectedLayer : public Layer {
protected:
  int inputFeatures;
  Matrix weights;

  // Matrix gradients;
  Matrix internal;
  Matrix outputs;

public:
  ConnectedLayer(int inputFeatures, const Layer &layer)
      : Layer(layer), inputFeatures(inputFeatures),
        weights(layer.getOuputFeatures(), inputFeatures),
        internal(layer.getOuputFeatures(), inputFeatures, false),
        outputs(layer.getOuputFeatures(), inputFeatures, false) {}
  ConnectedLayer(const Layer &a, const Layer &b)
      : ConnectedLayer(b.getOuputFeatures(), a) {}

  int getInputFeatures() const { return inputFeatures; }
  const Matrix &getWeights() const { return weights; }
};

class NeuralNetwork {
private:
  std::vector<ConnectedLayer> layers;

public:
  NeuralNetwork(int inputFeatures, std::vector<Layer> l) {
    // employ back
    layers.push_back(ConnectedLayer(inputFeatures, l[0]));
    for (size_t i = 1; i < l.size(); i++)
      layers.push_back(ConnectedLayer(l[i - 1].getOuputFeatures(), l[i]));
  }

  Matrix predict(Matrix inputs) {
    MatrixMath mm;
    std::vector<Matrix> steps;
    steps.push_back(inputs);
    for (size_t i = 0; i < layers.size(); i++) {
      Matrix internal = mm.mult(steps[steps.size() - 1], layers[i].getWeights(),
                                true, &layers[i].getBias());
      Matrix output = mm.activate(internal, layers[i].getActivation(),
                                  layers[i].getAlpha());
      steps.push_back(output);
    }
    mm.await();
    return steps[steps.size() - 1];
  }

  Matrix training(Matrix inputs) {
    MatrixMath mm;
    std::vector<Matrix> steps;
    steps.push_back(inputs);
    for (size_t i = 0; i < layers.size(); i++)
      steps.push_back(mm.mult(steps[steps.size() - 1], layers[i].getWeights(),
                              true, &layers[i].getBias(),
                              layers[i].getActivation(), layers[i].getAlpha()));
    mm.await();
    return steps[steps.size() - 1];
  }

  const Layer &getLayer(int i) const { return layers[i]; }
};

OpenCL openCL;

int main() {
  NeuralNetwork nn(
      2, {Layer(3, Activation::SIGMOID,
                Vector(std::vector<float>{0.0f, 0.0f, 0.0f})),
          Layer(1, Activation::SIGMOID, Vector(std::vector<float>{0.0f}))});

  for (int i = 0; i < 4; i++) {
    int v1 = (i / 2) % 2;
    int v2 = i % 2;

    Matrix input(1, 2, {static_cast<float>(v1), static_cast<float>(v2)});

    Matrix r = nn.predict(input);
    std::vector<float> rv = r.toVector();

    std::cout << "Network: ";
    for (size_t j = 0; j < rv.size(); ++j) {
      printf("%f\t", rv[j]);
    }

    float expected = static_cast<float>(v1 ^ v2);
    std::cout << " | XOR(" << v1 << ", " << v2 << ") = " << expected;

    std::cout << std::endl;
  }

  return 0;
}