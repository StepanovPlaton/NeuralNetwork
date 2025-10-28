__kernel void mult(__global float* A, __global float* B, __global float* C, 
                            int M, int N, int K) {
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
        
        int load_i_A = tile_offset + local_i;
        int load_j_A = tile_offset + local_j;
        
        if (global_i < M && load_j_A < K) {
            tile_A[local_j][local_i] = A[global_i * K + load_j_A];
        } else {
            tile_A[local_j][local_i] = 0.0f;
        }
        
        int load_i_B = tile_offset + local_i;
        int load_j_B = tile_offset + local_j;
        
        if (load_i_B < K && global_j < N) {
            tile_B[local_j][local_i] = B[load_i_B * N + global_j];
        } else {
            tile_B[local_j][local_i] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        #pragma unroll
        for (int k = 0; k < tile_size; k++) {
            sum += tile_A[k][local_i] * tile_B[local_j][k];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_i < M && global_j < N) {
        C[global_i * N + global_j] = sum;
    }
}

__kernel void mult_sc(__global float* A, __global float* B, float scalar, int M, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    B[i * N + j] = A[i * N + j] * scalar;
}

__kernel void add(__global float* A, __global float* B, __global float* C, float a, float b, int M, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    C[i * N + j] = (A[i * N + j] * a) + (B[i * N + j] * b);
}

__kernel void add_sc(__global float* A, __global float* B, float scalar, int M, int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    B[i * N + j] = A[i * N + j] + scalar;
}
