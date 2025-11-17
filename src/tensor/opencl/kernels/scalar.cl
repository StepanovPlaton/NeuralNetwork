__kernel void add(__global float *A, float scalar) {
  int i = get_global_id(0);
  A[i] += scalar;
}

__kernel void mult(__global float *A, float scalar) {
  int i = get_global_id(0);
  A[i] *= scalar;
}
