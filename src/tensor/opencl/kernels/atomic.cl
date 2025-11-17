__kernel void positive(__global float *A) {
  int i = get_global_id(0);
  A[i] = +A[i];
}

__kernel void negative(__global float *A) {
  int i = get_global_id(0);
  A[i] = -A[i];
}


float activate_x(float x, const int activation_type, const float alpha) {
  switch (activation_type) {
  case 0: // LINEAR
    return x;
  case 1: // SIGMOID
    return 1.0f / (1.0f + exp(-x));
  case 2: // TANH
    return tanh(x);
  case 3: // RELU
    return fmax(0.0f, x);
  case 4: // LEAKY_RELU
    return (x > 0.0f) ? x : alpha * x;
  case 5: // ELU
    return (x > 0.0f) ? x : alpha * (exp(x) - 1.0f);
  default:
    return x;
  }
}
__kernel void activate(__global float *input, __global float *output,
                       const int activation_type, const float alpha) {
  int i = get_global_id(0);
  output[i] = activate_x(input[i], activation_type, alpha);
}
