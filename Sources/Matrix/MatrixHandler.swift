import Metal

class MatrixHandler {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineMultiply: MTLComputePipelineState
    let pipelineHadamard: MTLComputePipelineState

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.commandQueue = commandQueue

        guard let library = try? device.makeLibrary(filepath: "metal/matrix_handler.metallib") else {
            fatalError("Metal Library not found")
        }

        guard let functionMultiply = library.makeFunction(name: "matrix_multiply") else {
            fatalError("Kernel: matrix_multiply not found")
        }
        self.pipelineMultiply = try! device.makeComputePipelineState(function: functionMultiply)

        guard let functionHadamard = library.makeFunction(name: "matrix_hadamard") else {
            fatalError("Kernel: matrix_hadamard not found")
        }
        self.pipelineHadamard = try! device.makeComputePipelineState(function: functionHadamard)
    }

    func matrixMultiply(matrixA: Matrix<Float>, matrixB: Matrix<Float>) -> [Float] {
        assert(matrixB.height == matrixA.width, "Matrix dimensions do not match for multiplication")

        let resultSize = matrixA.height * matrixB.width
        var result = [Float](repeating: 0.0, count: resultSize)

        let bufferA = device.makeBuffer(bytes: matrixA.elements1Dim, length: matrixA.count * MemoryLayout<Float>.size, options: [])
        let bufferB = device.makeBuffer(bytes: matrixB.elements1Dim, length: matrixB.count * MemoryLayout<Float>.size, options: [])
        let bufferResult = device.makeBuffer(bytes: &result, length: result.count * MemoryLayout<Float>.size, options: [])

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipelineMultiply)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferResult, offset: 0, index: 2)
        encoder.setBytes([Int32(matrixA.width)], length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes([Int32(matrixB.width)], length: MemoryLayout<Int32>.size, index: 4)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerGrid = MTLSize(width: matrixB.width, height: matrixA.height, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let output = bufferResult!.contents().bindMemory(to: Float.self, capacity: result.count)
        return Array(UnsafeBufferPointer(start: output, count: result.count))
    }

    func matrixHadamard(matrixA: [Float], matrixB: [Float], width: Int) -> [Float] {
        let height = matrixA.count / width
        assert(matrixA.count == matrixB.count, "Matrix dimensions do not for hadamard")
        let resultSize = matrixA.count
        var result = [Float](repeating: 0.0, count: resultSize)

        let bufferA = device.makeBuffer(bytes: matrixA, length: matrixA.count * MemoryLayout<Float>.size, options: [])
        let bufferB = device.makeBuffer(bytes: matrixB, length: matrixB.count * MemoryLayout<Float>.size, options: [])
        let bufferResult = device.makeBuffer(bytes: &result, length: result.count * MemoryLayout<Float>.size, options: [])

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipelineHadamard)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferResult, offset: 0, index: 2)
        encoder.setBytes([Int32(width)], length: MemoryLayout<Int32>.size, index: 3)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerGrid = MTLSize(width: width, height: height, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let output = bufferResult!.contents().bindMemory(to: Float.self, capacity: result.count)
        return Array(UnsafeBufferPointer(start: output, count: result.count))
    }
}