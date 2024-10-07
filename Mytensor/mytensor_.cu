#include <iostream>
#include <string>
#include <vector>
#include <math.h>

const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N)
{
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

inline double sigmoid(double n){
    return 1 / (1 + exp(-n));
}

class Tensor{
public:
    double *cpu_in;
    double *cpu_out;
    double *gpu_in;
    double *gpu_out;
    double *gpu_grid;
    int size;
    int ndim;
    std::vector<int> shape;
    std::vector<int> stride;
    std::string device;

    Tensor(std::vector<int> &sh, const std::string &de = "cpu"):shape(sh),ndim(sh.size()),
    device(de),size(1)
    {
        for (int i = 0; i < ndim;++i){
            stride.push_back(size);
            size *= shape[i];
        }
        cudaMalloc(&gpu_in, size * sizeof(double));
        cudaMalloc(&gpu_out, size * sizeof(double));
        cudaMalloc(&gpu_grid, sizeof(double) * size);
        cpu_in = new double[size];
        cpu_out = new double[size];
    }

    ~Tensor(){
        delete[] cpu_in;
        delete[] cpu_out;
        cudaFree(gpu_in);
        cudaFree(gpu_out);
        cudaFree(gpu_grid);
    }

    void toCpu(){
        cudaMemcpy(cpu_out, gpu_out, size * sizeof(double), cudaMemcpyDeviceToHost);
        device = "cpu";
    }

    void toGpu(){
        cudaMemcpy(gpu_in, cpu_in, size * sizeof(double), cudaMemcpyHostToDevice);
        device = "gpu";
    }

    Tensor(const Tensor& T){
        size = T.size;
        stride = T.stride;
        device = T.device;
        shape = T.shape;
        cudaMalloc(&gpu_in, size * sizeof(double));
        cudaMalloc(&gpu_out, size * sizeof(double));
        cpu_in = new double[size];
        cpu_out = new double[size];
        memcpy(cpu_in, T.cpu_in, sizeof(double) * size);
        memcpy(cpu_out, T.cpu_out, sizeof(double) * size);
        cudaMemcpy(gpu_in, T.gpu_in, sizeof(double) * size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpu_out, T.gpu_out, sizeof(double) * size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpu_grid, T.gpu_grid, sizeof(double) * size, cudaMemcpyDeviceToDevice);
    }

    Tensor cpu(){
        Tensor tt(*this);
        tt.toCpu();
        return tt;
    }

    Tensor gpu(){
        Tensor tt(*this);
        tt.toGpu();
        return tt;
    }
};


__global__ void reluForw(double* gpu_in, double* gpu_out, int size){
    CUDA_KERNEL_LOOP(i,size)
        gpu_out[i] = gpu_in[i] > 0 ? gpu_in[i] : 0;
}

__global__ void reluBack(double* gpu_in, double* gpu_out,double* gpu_grid, int size,double *l){
    //l为上一层的梯度
    CUDA_KERNEL_LOOP(i, size)
    gpu_grid[i] = gpu_in[i] > 0 ? l[i] : 0;
}

__global__ void sigmoidForw(double* gpu_in, double* gpu_out, int size){
    CUDA_KERNEL_LOOP(i, size)
    gpu_out[i] = sigmoid(gpu_in[i]);
}

__global__ void sigmoidBack(double* gpu_in, double* gpu_out, double* gpu_grid, int size,double *l){
    CUDA_KERNEL_LOOP(i,size)
        gpu_grid[i] = l[i] * (1 - gpu_out[i]) * gpu_out[i];
}