__kernel void add(__global float *A, __global float *B) {
  int i = get_global_id(0);
  A[i] += B[i];
}
__kernel void hadamard_mult(__global float *A, __global float *B) {
  int i = get_global_id(0);
  A[i] *= B[i];
}

#define TILE_SIZE 16
#define VEC_SIZE 4
__kernel void mult(__global float *A, __global float *B, __global float *C,
                   const int M, const int N, const int K) {
    
    const int row = get_global_id(0) * VEC_SIZE;
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    __local float tile_A[TILE_SIZE][TILE_SIZE + 1]; // +1 для избежания bank conflicts
    __local float tile_B[TILE_SIZE][TILE_SIZE + 1];
    
    float4 sum[VEC_SIZE];
    for (int i = 0; i < VEC_SIZE; i++) {
        sum[i] = (float4)(0.0f);
    }
    
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Загрузка tile_A с векторизацией
        int a_col = t * TILE_SIZE + local_col;
        #pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            int current_row = row + v;
            if (current_row < M && a_col < K) {
                tile_A[local_row * VEC_SIZE + v][local_col] = A[current_row * K + a_col];
            } else {
                tile_A[local_row * VEC_SIZE + v][local_col] = 0.0f;
            }
        }
        
        // Загрузка tile_B
        int b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            tile_B[local_row][local_col] = B[b_row * N + col];
        } else {
            tile_B[local_row][local_col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Векторизованное вычисление
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float4 a_vals = (float4)(
                tile_A[local_row * VEC_SIZE + 0][k],
                tile_A[local_row * VEC_SIZE + 1][k],
                tile_A[local_row * VEC_SIZE + 2][k],
                tile_A[local_row * VEC_SIZE + 3][k]
            );
            float b_val = tile_B[k][local_col];
            
            sum[0] += a_vals.x * b_val;
            sum[1] += a_vals.y * b_val;
            sum[2] += a_vals.z * b_val;
            sum[3] += a_vals.w * b_val;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Сохранение результатов с векторизацией
    #pragma unroll
    for (int v = 0; v < VEC_SIZE; v++) {
        int current_row = row + v;
        if (current_row < M && col < N) {
            C[current_row * N + col] = sum[v].x + sum[v].y + sum[v].z + sum[v].w;
        }
    }
}