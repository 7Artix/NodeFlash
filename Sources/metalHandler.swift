import Metal

class MetalHandler {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState

    init(kernelName: String, metalFileName: String) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal is not supported"])
        }
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        guard let library = try? device.makeLibrary(filepath: metalFileName) else {
            throw NSError(domain: "Metal", code: 2, userInfo: [NSLocalizedDescriptionKey: "Metal Library not found"])
        }
        guard let kernel = library.makeFunction(name: kernelName) else {
            throw NSError(domain: "Metal", code: 3, userInfo: [NSLocalizedDescriptionKey: "Kernel \(kernelName) not found"])
        }
        self.pipeline = try device.makeComputePipelineState(function: kernel)
    }

    func matrixMultiplication(matrixA: [Float], matrixB: [Float], widthA: Int, widthB: Int) -> [Float] {
        let heightA = matrixA.count / widthA
        let heightB = matrixB.count / widthB
        assert(heightB == widthA, "Matrix dimensions do not match for multiplication")

        let resultSize = heightA * widthB
        var result = [Float](repeating: 0, count: resultSize)

        let bufferA = device.makeBuffer(bytes: matrixA, length: matrixA.count * MemoryLayout<Float>.size, options: [])
        let bufferB = device.makeBuffer(bytes: matrixB, length: matrixB.count * MemoryLayout<Float>.size, options: [])
        let bufferResult = device.makeBuffer(bytes: &result, length: result.count * MemoryLayout<Float>.size, options: [])

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferResult, offset: 0, index: 2)
        encoder.setBytes([Int32(widthA)], length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes([Int32(widthB)], length: MemoryLayout<Int32>.size, index: 4)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerGrid = MTLSize(width: widthB, height: heightA, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let output = bufferResult!.contents().bindMemory(to: Float.self, capacity: result.count)
        return Array(UnsafeBufferPointer(start: output, count: result.count))
    }
}