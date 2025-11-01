#include "./math/math.hpp"

#include <chrono>
#include <thread>

using namespace GPU;

class Layer {
protected:
  int inputFeatures;
  int outputFeatures;
  Vector bias;
  Activation activation;
  float alpha;
  Matrix weights;

public:
  Layer(int inputFeatures, int outputFeatures, Activation activation,
        Vector bias, float alpha = 0.0f)
      : inputFeatures(inputFeatures), outputFeatures(outputFeatures),
        bias(bias), activation(activation), alpha(alpha),
        weights(outputFeatures, inputFeatures) {}

  int getInputFeatures() const { return inputFeatures; }
  int getOuputFeatures() const { return outputFeatures; }
  Activation getActivation() const { return activation; }
  float getAlpha() const { return alpha; }

  const Vector &getBias() const { return bias; }
  const Matrix &getWeights() const { return weights; }
};

class NeuralNetwork {
private:
  std::vector<Layer> layers;

public:
  NeuralNetwork(std::vector<Layer> l) : layers(l) {}

  Matrix predict(Matrix inputs) {
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
      {Layer(2, 1, Activation::SIGMOID, Vector(std::vector<float>{1.0f}))});

  for (int i = 0; i < 10; i++) {
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