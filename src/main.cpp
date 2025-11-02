#define NOGPU

#include "math/math.hpp"

#ifdef NOGPU
using namespace CPU;
#else
using namespace GPU;
#endif

class Layer {
protected:
  int outputFeatures;
  Vector bias;
  Activation activation;
  float alpha;

public:
  Layer(int outputFeatures, Activation activation, float alpha = 0.0f)
      : outputFeatures(outputFeatures), bias(outputFeatures),
        activation(activation), alpha(alpha) {}

  int getOuputFeatures() const { return outputFeatures; }
  Activation getActivation() const { return activation; }
  float getAlpha() const { return alpha; }

  const Vector &getBias() const { return bias; }
};

class ConnectedLayer : public Layer {
protected:
  int inputFeatures;
  Matrix weights;

public:
  ConnectedLayer(int inputFeatures, const Layer &layer)
      : Layer(layer), inputFeatures(inputFeatures),
        weights(layer.getOuputFeatures(), inputFeatures) {}
  ConnectedLayer(const Layer &a, const Layer &b)
      : ConnectedLayer(b.getOuputFeatures(), a) {}

  int getInputFeatures() const { return inputFeatures; }
  const Matrix &getWeights() const { return weights; }
};

class LearnLayer : public ConnectedLayer {
protected:
  // Matrix gradients;
  Matrix internal;
  Matrix outputs;

public:
  LearnLayer(int inputFeatures, const Layer &layer)
      : ConnectedLayer(inputFeatures, layer),
        internal(layer.getOuputFeatures(), inputFeatures, false),
        outputs(layer.getOuputFeatures(), inputFeatures, false) {}
  LearnLayer(const Layer &a, const Layer &b)
      : LearnLayer(b.getOuputFeatures(), a) {}

  const Matrix &getInternal() const { return internal; }
  const Matrix &getOutputs() const { return outputs; }
  void setInternal(const Matrix &i) { internal = i; }
  void setOutputs(const Matrix &o) { outputs = o; }
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
    for (size_t i = 0; i < layers.size(); i++)
      steps.push_back(mm.dot(steps[steps.size() - 1], layers[i].getWeights(),
                             true, &layers[i].getBias(),
                             layers[i].getActivation(), layers[i].getAlpha()));
    mm.await();
    return steps[steps.size() - 1];
  }

  const ConnectedLayer &getLayer(int i) const { return layers[i]; }
};

class LearnNerualNetrowk {
private:
  std::vector<LearnLayer> layers;

public:
  LearnNerualNetrowk(int inputFeatures, std::vector<Layer> l) {
    // employ back
    layers.push_back(LearnLayer(inputFeatures, l[0]));
    for (size_t i = 1; i < l.size(); i++)
      layers.push_back(LearnLayer(l[i - 1].getOuputFeatures(), l[i]));
  }

  Matrix learn(Matrix inputs, Matrix target) {
    MatrixMath mm;
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i].setInternal(mm.dot(i == 0 ? inputs : layers[i - 1].getOutputs(),
                                   layers[i].getWeights(), true,
                                   &layers[i].getBias()));
      layers[i].setOutputs(mm.activate(layers[i].getInternal(),
                                       layers[i].getActivation(),
                                       layers[i].getAlpha()));
    }
    mm.await();

    std::vector<float> io = inputs.toVector();
    std::cout << "I: ";
    for (size_t i = 0; i < io.size(); ++i)
      printf("%5.3f ", io[i]);
    std::vector<float> no = layers[layers.size() - 1].getOutputs().toVector();
    std::cout << "| NN: ";
    for (size_t i = 0; i < no.size(); ++i)
      printf("%5.3f ", no[i]);
    std::vector<float> to = target.toVector();
    std::cout << "| T: ";
    for (size_t i = 0; i < to.size(); ++i)
      printf("%5.3f ", to[i]);
    Matrix mse =
        mm.loss(layers[layers.size() - 1].getOutputs(), target, Loss::MSE);
    std::vector<float> lo = mse.toVector();
    std::cout << "| L: ";
    for (size_t i = 0; i < lo.size(); ++i)
      printf("%5.3f ", lo[i]);
    std::cout << std::endl;

    // Matrix dA2 =
    //     mm.d_loss(layers[layers.size() - 1].getOutputs(), target, Loss::MSE);
    // Matrix  = mm.dot(dA2,
    // mm.d_activate(layers[layers.size()-1].getOutputs()));

    return mse;
  }

  const LearnLayer &getLayer(int i) const { return layers[i]; }
};

#ifndef NOGPU
OpenCL openCL;
#endif

int main() {
  LearnNerualNetrowk nn(
      2, {Layer(3, Activation::SIGMOID), Layer(3, Activation::SIGMOID)});
  std::cout << "NN created!" << std::endl;

  for (int i = 0; i < 4; i++) {
    int v1 = (i / 2) % 2;
    int v2 = i % 2;

    Matrix input(1, 2, {static_cast<float>(v1), static_cast<float>(v2)});
    Matrix target(1, 3,
                  {static_cast<float>(v1 ^ v2), static_cast<float>(v1 & v2),
                   static_cast<float>(v1 | v2)});

    nn.learn(input, target);
  }

  return 0;
}
