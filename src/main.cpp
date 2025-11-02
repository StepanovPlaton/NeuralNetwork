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
  void setBias(const Vector &b) { bias = b; }
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
  void setWeights(const Matrix &w) { weights = w; }
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
                             false, true, &layers[i].getBias(),
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

  Matrix learn(Matrix inputs, Matrix target, float speed = 1.0f) {
    MatrixMath mm;
    VectorMath vm;
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i].setInternal(mm.dot(i == 0 ? inputs : layers[i - 1].getOutputs(),
                                   layers[i].getWeights(), false, true,
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

    Matrix dAnl =
        mm.d_loss(layers[layers.size() - 1].getOutputs(), target, Loss::MSE);
    for (int i = layers.size() - 1; i >= 0; --i) {
      Matrix dZl = mm.mult(dAnl, mm.d_activate(layers[i].getInternal()));
      Matrix dWl = mm.mult(
          mm.dot(dZl, i == 0 ? inputs : layers[i - 1].getOutputs(), true),
          1.0f / (float)inputs.getRows());
      Vector dbl = mm.axis_sum(mm.mult(dZl, 1.0f / (float)inputs.getRows()));
      dAnl = mm.dot(dZl, layers[i].getWeights(), false, false); // false true?!

      mm.await();

      layers[i].setWeights(mm.add(layers[i].getWeights(), dWl, -speed));
      layers[i].setBias(
          vm.add(layers[i].getBias(), dbl, -speed / (float)inputs.getRows()));
    }

    return mse;
  }

  const LearnLayer &getLayer(int i) const { return layers[i]; }
};

#ifndef NOGPU
OpenCL openCL;
#endif

int main() {
  LearnNerualNetrowk nn(
      2, {Layer(2, Activation::TANH), Layer(1, Activation::SIGMOID)});
  std::cout << std::endl;

  // Matrix input(4, 2);
  // Matrix target(4, 1);
  //
  // for (int batch = 0; batch < 4; batch++) {
  //   for (int i = 0; i < 4; i++) {
  //     int v1 = (i / 2) % 2;
  //     int v2 = i % 2;
  //
  //     input(i, 0) = static_cast<float>(v1);
  //     input(i, 1) = static_cast<float>(v2);
  //     target(i, 0) = static_cast<float>(v1 ^ v2);
  //   }
  // }
  //
  // for (int i = 0; i < 10; i++) {
  //   printf("%4d | ", i + 1);
  //   Matrix mse = nn.learn(input, target, 0.1f * std::pow(0.99, i));
  // }

  for (int i = 0; i < 4 * 1000; i++) {
    int v1 = (i / 2) % 2;
    int v2 = i % 2;

    Matrix input(1, 2, {static_cast<float>(v1), static_cast<float>(v2)});
    Matrix target(1, 1, static_cast<float>(v1 ^ v2));

    printf("%5d | ", i + 1);
    Matrix mse = nn.learn(input, target, 0.00003f);
    if (i % 4 == 3)
      std::cout << std::endl;
  }

  return 0;
}
