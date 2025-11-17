__kernel void add(__global float *A, __global float *B, __global float *C,
                  float x) {
  int i = get_global_id(0);
  C[i] = A[i] + (B[i] * x);
}
__kernel void mult(__global float *A, __global float *B, __global float *C,
                  float x) {
  int i = get_global_id(0);
  C[i] = A[i] * (B[i] * x);
}

float activate(float x, const int activation_type, const float alpha) {
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

__kernel void mult_small(__global float *A, __global float *B,
                         __global float *C, __global float *bias,
                         const int activation_type, const float alpha,
                         const int M, const int N, const int K,
                         const int transpose_B) {
  const int row = get_global_id(0);
  const int col = get_global_id(1);

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      float a_val = A[row * K + k];

      float b_val;
      if (transpose_B) {
        b_val = B[col * K + k];
      } else {
        b_val = B[k * N + col];
      }

      sum += a_val * b_val;
    }

    float result = sum + bias[col];
    if (activation_type != 0) {
      result = activate(result, activation_type, alpha);
    }
    C[row * N + col] = result;
  }
}

__kernel void mult(__global float *A, __global float *B, __global float *C,
                   __global float *bias, const int activation_type,
                   const float alpha, const int M, const int N, const int K,
                   const int transpose_B) {
  const int tile_size = 16;

  int local_i = get_local_id(0);
  int local_j = get_local_id(1);
  int local_size_i = get_local_size(0);
  int local_size_j = get_local_size(1);

  int global_i = get_group_id(0) * local_size_i + local_i;
  int global_j = get_group_id(1) * local_size_j + local_j;

  __local float tile_A[16][16];
  __local float tile_B[16][16];

  float sum = 0.0f;

  int num_tiles = (K + tile_size - 1) / tile_size;

  for (int tile = 0; tile < num_tiles; tile++) {
    int tile_offset = tile * tile_size;

    // Загрузка tile_A (без изменений)
    int load_i_A = tile_offset + local_i;
    int load_j_A = tile_offset + local_j;

    if (global_i < M && load_j_A < K) {
      tile_A[local_j][local_i] = A[global_i * K + load_j_A];
    } else {
      tile_A[local_j][local_i] = 0.0f;
    }

    // Загрузка tile_B с учетом транспонирования
    int load_i_B = tile_offset + local_i;
    int load_j_B = tile_offset + local_j;

    if (transpose_B) {
      // B транспонирована: обращаем индексы
      if (load_i_B < N && global_j < K) {
        tile_B[local_j][local_i] = B[global_j * N + load_i_B];
      } else {
        tile_B[local_j][local_i] = 0.0f;
      }
    } else {
      // B не транспонирована (оригинальная логика)
      if (load_i_B < K && global_j < N) {
        tile_B[local_j][local_i] = B[load_i_B * N + global_j];
      } else {
        tile_B[local_j][local_i] = 0.0f;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int k = 0; k < tile_size; ++k) {
      sum += tile_A[k][local_i] * tile_B[local_j][k];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (global_i < M && global_j < N) {
    float result = sum + bias[global_j];
    if (activation_type != 0) {
      result = activate(result, activation_type, alpha);
    }
    C[global_i * N + global_j] = result;
  }
}

