__kernel void add(__global float *A, __global float *B, float scalar) {
  int i = get_global_id(0);
  B[i] = A[i] + scalar;
}

__kernel void mult(__global float *A, __global float *B, float scalar) {
  int i = get_global_id(0);
  B[i] = A[i] * scalar;
}
