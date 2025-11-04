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
      : ConnectedLayer(a.getOuputFeatures(), b) {}

  int getInputFeatures() const { return inputFeatures; }
  const Matrix &getWeights() const { return weights; }
  void setWeights(const Matrix &w) { weights = w; }
};

class LearnLayer : public ConnectedLayer {
protected:
  Matrix internal;
  Matrix outputs;

public:
  LearnLayer(int inputFeatures, const Layer &layer)
      : ConnectedLayer(inputFeatures, layer),
        internal(layer.getOuputFeatures(), inputFeatures, false),
        outputs(layer.getOuputFeatures(), inputFeatures, false) {}
  LearnLayer(const Layer &a, const Layer &b)
      : LearnLayer(a.getOuputFeatures(), b) {}

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
      layers.push_back(LearnLayer(l[i - 1], l[i]));
  }

  Matrix learn(Matrix inputs, Matrix target, float speed = 1.0f) {
    MatrixMath mm;
    VectorMath vm;
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i].setInternal(mm.dot(layers[i].getWeights(),
                                   i == 0 ? inputs : layers[i - 1].getOutputs(),
                                   false, false, &layers[i].getBias()));
      layers[i].setOutputs(mm.activate(layers[i].getInternal(),
                                       layers[i].getActivation(),
                                       layers[i].getAlpha()));
    }
    mm.await();

    std::vector<float> io = inputs.toVector();
    std::cout << "I: ";
    for (size_t i = 0; i < io.size(); ++i)
      printf("%4.2f ", io[i]);

    std::vector<float> ni = layers[layers.size() - 1].getInternal().toVector();
    std::cout << "| NNI: ";
    for (size_t i = 0; i < ni.size(); ++i)
      printf("%4.2f ", ni[i]);

    std::vector<float> no = layers[layers.size() - 1].getOutputs().toVector();
    std::cout << "| NNO: ";
    for (size_t i = 0; i < no.size(); ++i)
      printf("%4.2f ", no[i]);

    std::vector<float> to = target.toVector();
    std::cout << "| T: ";
    for (size_t i = 0; i < to.size(); ++i)
      printf("%4.2f ", to[i]);

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
      printf("=== Layer %d ===\n", i + 1);
      printf("dAnl: ");
      dAnl.print();

      Matrix dZl = mm.mult(dAnl, mm.d_activate(layers[i].getInternal()));
      printf("dZl: ");
      dZl.print();

      Matrix dWl =
          mm.mult(mm.dot(dZl, i == 0 ? inputs : layers[i - 1].getOutputs(),
                         false, true),
                  1.0f / (float)inputs.getRows());
      printf("dWl: ");
      dWl.print();

      Vector dbl = mm.axis_sum(mm.mult(dZl, 1.0f / (float)inputs.getRows()));
      printf("dbl: ");
      dbl.print();

      dAnl = mm.dot(layers[i].getWeights(), dZl, true); // false true?!

      mm.await();

      layers[i].setWeights(mm.add(layers[i].getWeights(), dWl, -speed));
      printf("Weights %d: ", i + 1);
      layers[i].getWeights().print();

      layers[i].setBias(
          vm.add(layers[i].getBias(), dbl, -speed / (float)inputs.getRows()));
      printf("Bias %d: ", i + 1);
      layers[i].getBias().print();
    }

    return mse;
  }

  const LearnLayer &getLayer(int i) const { return layers[i]; }

  // delete
  LearnLayer &getLayer(int i) { return layers[i]; }
};

#ifndef NOGPU
OpenCL openCL;
#endif

int main() {
  // LearnNerualNetrowk nn(
  //     3, {Layer(3, Activation::SIGMOID), Layer(3, Activation::SIGMOID)});
  //
  // Matrix weights1(3, 3,
  //                 {0.88f, 0.39f, 0.9f, 0.37f, 0.14f, 0.41f, 0.96f, 0.5f,
  //                 0.6f});
  // Matrix weights2(
  //     3, 3, {0.29f, 0.57f, 0.36f, 0.73f, 0.53f, 0.68f, 0.01f, 0.02f, 0.58f});
  //
  // Vector bias1(std::vector<float>{0.23f, 0.89f, 0.08f});
  // Vector bias2(std::vector<float>{0.78f, 0.83f, 0.8f});
  //
  // nn.getLayer(0).setWeights(weights1);
  // nn.getLayer(0).setBias(bias1);
  //
  // nn.getLayer(1).setWeights(weights2);
  // nn.getLayer(1).setBias(bias2);
  //
  // std::cout << std::endl;
  //
  // Matrix input(3, 1, {0.03f, 0.72f, 0.49f});
  // Matrix target(3, 1, {0.93f, 0.74f, 0.17f});
  //
  // // for (int i = 0; i < 1000; i++)
  // nn.learn(input, target, 0.01f);

  LearnNerualNetrowk nn(
      2, {Layer(3, Activation::SIGMOID), Layer(1, Activation::SIGMOID)});

  Matrix input(2, 4);
  Matrix target(1, 4);

  float min = 100.0f;
  for (int batch = 0; batch < 4; batch++) {
    for (int i = 0; i < 4; i++) {
      int v1 = (i / 2) % 2;
      int v2 = i % 2;

      input(0, i) = static_cast<float>(v1);
      input(1, i) = static_cast<float>(v2);
      target(0, i) = static_cast<float>(v1 ^ v2);
    }
  }

  for (int i = 0; i < 1000; i++) {
    printf("%4d | ", i + 1);
    Matrix mse = nn.learn(input, target, 0.0001f * std::pow(0.99f, i));
    std::vector<float> lv = mse.toVector();
    float loss = 0.0f;
    for (size_t i = 0; i < lv.size(); ++i)
      loss += lv[i];
    if (loss < min)
      min = loss;
  }
  std::cout << min << std::endl;

  // LearnNerualNetrowk nn(
  //     2, {Layer(3, Activation::SIGMOID), Layer(1, Activation::SIGMOID)});
  // float min = 100.0f;
  // for (int i = 0; i < 4 * 10000; i++) {
  //   int v1 = (i / 2) % 2;
  //   int v2 = i % 2;
  //
  //   Matrix input(2, 1, {static_cast<float>(v1), static_cast<float>(v2)});
  //   Matrix target(1, 1, static_cast<float>(v1 ^ v2));
  //
  //   printf("%5d | ", i + 1);
  //   Matrix mse = nn.learn(input, target, 0.0001f * std::pow(0.95f, i));
  //   if (i % 4 == 3)
  //     std::cout << std::endl;
  //   if (mse[0] < min)
  //     min = mse[0];
  // }
  // std::cout << min << std::endl;

  return 0;
}
