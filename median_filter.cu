#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

#define BLOCK_SIZE 16

// Bitonic sort kernel
__device__ void bitonicMerge(float* data, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        for (int i = low; i < low + k; i++) {
            if ((data[i] > data[i + k]) == dir) {
                // Swap
                float temp = data[i];
                data[i] = data[i + k];
                data[i + k] = temp;
            }
        }
        bitonicMerge(data, low, k, dir);
        bitonicMerge(data, low + k, k, dir);
    }
}

__device__ void bitonicSort(float* data, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        // Sort in ascending order
        bitonicSort(data, low, k, 1);
        // Sort in descending order
        bitonicSort(data, low + k, k, 0);
        // Merge the whole sequence in the direction of sorting
        bitonicMerge(data, low, cnt, dir);
    }
}

__global__ void medianFilter(const float* input, float* output, int width, int height, int windowSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfWindow = windowSize / 2;

    if (x < width && y < height) {
        int windowElements = windowSize * windowSize;
        float* window = new float[windowElements]; // Allocate dynamic window size

        int idx = 0;
        for (int i = -halfWindow; i <= halfWindow; ++i) {
            for (int j = -halfWindow; j <= halfWindow; ++j) {
                int ix = min(max(x + i, 0), width - 1);
                int iy = min(max(y + j, 0), height - 1);
                window[idx++] = input[iy * width + ix];
            }
        }

        // Sort the window using bitonic sort
        bitonicSort(window, 0, windowElements, 1); // 1 for ascending order
        output[y * width + x] = window[windowElements / 2]; // Get median
        delete[] window; // Free the allocated memory
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path> <output_image_path> <window_size>" << std::endl;
        return -1;
    }

    std::string inputImagePath = argv[1];
    std::string outputImagePath = argv[2];
    int windowSize = std::stoi(argv[3]);

    // Load the image in color
    cv::Mat img = cv::imread(inputImagePath, cv::IMREAD_COLOR);//also creates channel if required
    if (img.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    int imageSize = width * height * sizeof(float);

    // Prepare the input and output arrays
    float* h_input = new float[width * height * channels];
    float* h_output = new float[width * height * channels];

    // Split the image into three channels
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(img, bgrChannels);

    // Allocate memory on the device for each channel
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Process each channel separately
    for (int c = 0; c < channels; ++c) {
        // Convert channel to float and normalize
        bgrChannels[c].convertTo(bgrChannels[c], CV_32F, 1.0 / 255.0);

        // Copy input channel to host memory
        memcpy(h_input, bgrChannels[c].ptr<float>(), imageSize);

        // Copy input channel to device memory
        cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

        // Set up execution configuration with fixed block and grid size
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Launch the median filter kernel with the dynamic window size
        medianFilter<<<gridDim, blockDim>>>(d_input, d_output, width, height, windowSize);
        cudaDeviceSynchronize();

        // Copy output back to host memory
        cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

        // Copy the filtered output back to the channel matrix
        memcpy(bgrChannels[c].ptr<float>(), h_output, imageSize);
        
        // Convert the filtered output back to [0, 255] range
        bgrChannels[c].convertTo(bgrChannels[c], CV_8U, 255.0);
    }

    // Merge the processed channels back into one image
    cv::Mat imgOutput;
    cv::merge(bgrChannels, imgOutput);

    // Save the processed image
    cv::imwrite(outputImagePath, imgOutput);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    std::cout << "Filtering complete. Output saved as '" << outputImagePath << "'." << std::endl;
    return 0;
}
