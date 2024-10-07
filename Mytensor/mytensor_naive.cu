#include <iostream>
#include <string>
#include <vector>

// Use 512 or 256 threads per block
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

__global__ void reluGpu(double *in,double* out,int n){
        printf("success!");
}

class Tensor
{
public:
    double *data; // 假设所有数据都是double类型
    int size;
    int ndim;
    std::vector<int> stride; // 步长
    std::vector<int> shape;
    std::string device; // 设备类型

    Tensor(std::vector<int> &sh, const std::string &de = "cpu")
    {
        size = sh[0] > 0;
        device = de;
        shape = sh;
        ndim = sh.size();
        for (int i = 0; i < ndim; ++i)
        {
            stride.push_back(size);
            size *= sh[i];
        }
        data = new double[size];
    }

    void debug()
    {
        for (int i = 0; i < ndim; ++i)
        {
            std::cout << stride[i] << ' ';
        }
    }

    ~Tensor(){
        delete[] data;
    }

    Tensor(const Tensor& T){
        size = T.size;
        ndim = T.ndim;
        data = new double[size];
        for (int i = 0; i < size;++i)
            data[i] = T.data[i];
        stride = T.stride;
        shape = T.shape;
        device = T.device;
    }

    Tensor cpu(){
        Tensor a(*this);
        a.device = "cpu";
        return a;
    }

    Tensor gpu(){
        Tensor a(*this);
        a.device = "gpu";
        return a;
    }
    
    void relu(double *in,double *out,int n){
        if(device=="cpu")
            for (int i = 0; i < n;++i)
                out[i] = in[i] > 0 ? in[i] : 0;
        else{

        }
    }

    void sigmoid(double* in,double *out,int n){
        if(device=="cpu"){
            
        }
    }
};  

int main()
{
    std::vector<int> s = {1, 2, 4};
    Tensor m(s, "gpu");
    double a = 1, b = 2, n = 1;
    reluGpu<<<1, 8>>>(&a, &b, n);
    cudaDeviceSynchronize();
}