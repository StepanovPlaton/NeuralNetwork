float activate_x(float x, const int activation_type, const float alpha) {
    switch(activation_type) {
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
        case 6: // GELU
            return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
        default:
            return x;
    }
}

__kernel void activate(
    __global float* input,
    __global float* output,
    const int activation_type,
    const float alpha,
    const int rows,
    const int cols)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        output[idx] = activate_x(input[idx], activation_type, alpha);
    }
}

__kernel void mult(
    __global float* A, 
    __global float* B, 
    __global float* C,
    const float bias,          
    const int activation_type, 
    const float alpha,         
    const int M, 
    const int N, 
    const int K) 
{
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
        float result = sum + bias; 
        if (activation_type != 0) {
            result = activate_x(result, activation_type, alpha);
        }
        C[global_i * N + global_j] = result;
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

