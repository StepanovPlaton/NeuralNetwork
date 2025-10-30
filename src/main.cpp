#include "./math/math.hpp"

#include <chrono>
#include <thread>

typedef Matrices::GPU M;
typedef MutableMatrices::GPU MM;

class Layer {
protected:
  int features;
  float bias;
  MM::Activate activate;
  float alpha;

public:
  Layer(int features, MM::Activate activate = MM::Activate::LINEAR,
        float bias = 0.0f, float alpha = 0.0f)
      : features(features), activate(activate), bias(bias), alpha(alpha) {}

  int getFeatures() const { return features; }
  float getBias() const { return bias; }
  MM::Activate getActivate() const { return activate; }
  float getAlpha() const { return alpha; }
};

class NeuralNetwork {
private:
  std::vector<Layer> layers;
  std::vector<MM> weights;

public:
  NeuralNetwork(int n, std::initializer_list<Layer> l) : layers(l) {
    weights.emplace_back(n, layers[0].getFeatures());
    for (int i = 0; i < layers.size() - 1; i++)
      weights.emplace_back(layers[i].getFeatures(),
                           layers[i + 1].getFeatures());
  }

  std::vector<float> predict(std::vector<float> i) {
    if (i.size() != weights[0].getRows())
      std::invalid_argument("Invalid input size");
    MM input(1, (int)i.size(), i);
    for (size_t i = 0; i < weights.size(); i++)
      input.mult(weights[i], layers[i + 1].getBias(),
                 layers[i + 1].getActivate(), layers[i + 1].getAlpha());
    return input.toVector();
  }
};

OpenCL openCL;

int main() {
  NeuralNetwork nn(
      2, {Layer(3, MM::Activate::RELU), Layer(1, MM::Activate::RELU)});

  for (int i = 0; i < 10; i++) {
    int v1 = (i / 2) % 2;
    int v2 = i % 2;

    std::vector<float> v = {static_cast<float>(v1), static_cast<float>(v2)};

    std::vector<float> r = nn.predict(v);
    float expected = static_cast<float>(v1 ^ v2);

    std::cout << "XOR(" << v1 << ", " << v2 << ") = " << expected;
    std::cout << " | Network: ";
    for (size_t j = 0; j < r.size(); ++j) {
      std::cout << r[j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}