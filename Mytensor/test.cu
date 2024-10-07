#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    helloFromGPU<<<1, 8>>>();
    cudaDeviceSynchronize(); // 等待 GPU 完成
    return 0;
}
