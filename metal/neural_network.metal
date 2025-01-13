#include <metal_stdlib>
using namespace metal;

kernel void forward_propagation(
    device const float* matrixA [[ buffer(0) ]],
    device const float* matrixB [[ buffer(1) ]],
    device float* result [[ buffer(2) ]],
    constant int& widthA [[ buffer(3) ]],
    constant int& heightA [[ buffer(4) ]],
    constant int& widthB [[ buffer(5) ]],
    uint2 id [[ thread_position_in_grid ]]
) {
    int row = id.y;
    int col = id.x;

    if (row < heightA && col < widthB) {
        float value = 0.0;
        for (int k = 0; k < widthA; k++) {
            value += matrixA[row * widthA + k] * matrixB[k * widthB + col];
        }
        result[row * widthB + col] = value;
    }
}