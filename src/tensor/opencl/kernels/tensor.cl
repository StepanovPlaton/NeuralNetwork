__kernel void add(__global float *A, __global float *B) {
  int i = get_global_id(0);
  A[i] += B[i];
}
__kernel void hadamard_mult(__global float *A, __global float *B) {
  int i = get_global_id(0);
  A[i] *= B[i];
}

#define TILE_SIZE 16
__kernel void mult(__global float *A, __global float *B, __global float *C,
                   const int M, const int N, const int K) {
    
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    __local float tile_A[TILE_SIZE][TILE_SIZE];
    __local float tile_B[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K - 1) / TILE_SIZE + 1; t++) {
        
        int a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            tile_A[local_row][local_col] = A[row * K + a_col];
        } else {
            tile_A[local_row][local_col] = 0.0f;
        }
        
        int b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            tile_B[local_row][local_col] = B[b_row * N + col];
        } else {
            tile_B[local_row][local_col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int k_max = min(TILE_SIZE, K - t * TILE_SIZE);
        for (int k = 0; k < k_max; k++) {
            sum += tile_A[local_row][k] * tile_B[k][local_col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

