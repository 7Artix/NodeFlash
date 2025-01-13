import Metal

class MatrixHandler {
    private let concurrentQueue = DispatchQueue(label: "com.example.metal.concurrent", attributes: .concurrent)
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipelineMultiply: MTLComputePipelineState
    let pipelineElementWise: MTLComputePipelineState
    let pipelineTranspose: MTLComputePipelineState

    private var bufferMatrixA: MTLBuffer?
    private var bufferMatrixB: MTLBuffer?
    private var bufferResult: MTLBuffer?

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

        guard let functionElementWise = library.makeFunction(name: "matrix_element_wise") else {
            fatalError("Kernel: matrix_element_wise not found")
        }
        self.pipelineElementWise = try! device.makeComputePipelineState(function: functionElementWise)

        guard let functionTranspose = library.makeFunction(name: "matrix_transpose") else {
            fatalError("Kernel: matrix_transpose not found")
        }
        self.pipelineTranspose = try! device.makeComputePipelineState(function: functionTranspose)
    }

    private func prepareBuffers(matrixA: Matrix<Float>, matrixB: Matrix<Float>, resultSize: Int) {
        if bufferMatrixA == nil || bufferMatrixA!.length < matrixA.count * MemoryLayout<Float>.size {
            bufferMatrixA = device.makeBuffer(length: matrixA.count * MemoryLayout<Float>.size, options: [])
        }
        if bufferMatrixB == nil || bufferMatrixB!.length < matrixB.count * MemoryLayout<Float>.size {
            bufferMatrixB = device.makeBuffer(length: matrixB.count * MemoryLayout<Float>.size, options: [])
        }
        if bufferResult == nil || bufferResult!.length < resultSize * MemoryLayout<Float>.size {
            bufferResult = device.makeBuffer(length: resultSize * MemoryLayout<Float>.size, options: [])
        }
        let ptrA = bufferMatrixA!.contents().bindMemory(to: Float.self, capacity: matrixA.count)
        ptrA.update(from: matrixA.elements1Dim, count: matrixA.count)
        bufferMatrixA!.didModifyRange(0..<matrixA.count * MemoryLayout<Float>.size)
        let ptrB = bufferMatrixB!.contents().bindMemory(to: Float.self, capacity: matrixB.count)
        ptrB.update(from: matrixB.elements1Dim, count: matrixB.count)
        bufferMatrixB!.didModifyRange(0..<matrixB.count * MemoryLayout<Float>.size)
    }

    private func prepareBuffers(matrixA: Matrix<Float>, resultSize: Int) {
        if bufferMatrixA == nil || bufferMatrixA!.length < matrixA.count * MemoryLayout<Float>.size {
            bufferMatrixA = device.makeBuffer(length: matrixA.count * MemoryLayout<Float>.size, options: [])
        }
        if bufferResult == nil || bufferResult!.length < resultSize * MemoryLayout<Float>.size {
            bufferResult = device.makeBuffer(length: resultSize * MemoryLayout<Float>.size, options: [])
        }
        let ptrA = bufferMatrixA!.contents().bindMemory(to: Float.self, capacity: matrixA.count)
        ptrA.update(from: matrixA.elements1Dim, count: matrixA.count)
        bufferMatrixA!.didModifyRange(0..<matrixA.count * MemoryLayout<Float>.size)
    }

    func matrixMultiply(matrixA: Matrix<Float>, matrixB: Matrix<Float>) -> Matrix<Float> {
        assert(matrixB.height == matrixA.width, "Matrix dimensions do not match for multiplication")

        let resultSize = matrixA.height * matrixB.width

        self.prepareBuffers(matrixA: matrixA, matrixB: matrixB, resultSize: resultSize)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipelineMultiply)
        encoder.setBuffer(bufferMatrixA, offset: 0, index: 0)
        encoder.setBuffer(bufferMatrixB, offset: 0, index: 1)
        encoder.setBuffer(bufferResult, offset: 0, index: 2)
        encoder.setBytes([Int32(matrixA.width)], length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes([Int32(matrixA.height)], length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes([Int32(matrixB.width)], length: MemoryLayout<Int32>.size, index: 5)

        let threadsPerThreadgroupWidth = min(matrixB.width, 32)
        let threadsPerThreadgroupHeight = min(matrixA.height, 32)
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupWidth, height: threadsPerThreadgroupHeight, depth: 1)

        let threadGroupsWidth = Int(ceil(Double(matrixB.width) / Double(threadsPerThreadgroup.width)))
        let threadGroupsHeight = Int(ceil(Double(matrixA.height) / Double(threadsPerThreadgroup.height)))
        let threadGroups = MTLSize(width: threadGroupsWidth, height: threadGroupsHeight, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let output = bufferResult!.contents().bindMemory(to: Float.self, capacity: resultSize)
        let matrixResult = Matrix<Float>(elements: Array(UnsafeBufferPointer(start: output, count: resultSize)), height: matrixA.height)
        return matrixResult
    }

    func matrixElementWise(matrixA: Matrix<Float>, matrixB: Matrix<Float>) -> Matrix<Float> {
        assert(matrixA.height == matrixB.height || matrixA.width == matrixB.width, "Matrix dimensions do not for hadamard")
        let resultSize = matrixA.count

        prepareBuffers(matrixA: matrixA, matrixB: matrixB, resultSize: resultSize)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipelineElementWise)
        encoder.setBuffer(bufferMatrixA, offset: 0, index: 0)
        encoder.setBuffer(bufferMatrixB, offset: 0, index: 1)
        encoder.setBuffer(bufferResult, offset: 0, index: 2)
        encoder.setBytes([Int32(matrixA.width)], length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes([Int32(matrixA.height)], length: MemoryLayout<Int32>.size, index: 4)

        let threadsPerThreadgroupWidth = min(matrixA.width, 32)
        let threadsPerThreadgroupHeight = min(matrixA.height, 32)
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupWidth, height: threadsPerThreadgroupHeight, depth: 1)

        let threadGroupsWidth = Int(ceil(Double(matrixA.width) / Double(threadsPerThreadgroup.width)))
        let threadGroupsHeight = Int(ceil(Double(matrixA.height) / Double(threadsPerThreadgroup.height)))
        let threadGroups = MTLSize(width: threadGroupsWidth, height: threadGroupsHeight, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let output = bufferResult!.contents().bindMemory(to: Float.self, capacity: resultSize)
        let matrixResult = Matrix<Float>(elements: Array(UnsafeBufferPointer(start: output, count: resultSize)), height: matrixA.height)
        return matrixResult
    }

    func matrixTranspose(matrixA: Matrix<Float>) -> Matrix<Float> {
        let resultSize = matrixA.count

        prepareBuffers(matrixA: matrixA, resultSize: resultSize)

        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!

        encoder.setComputePipelineState(pipelineTranspose)
        encoder.setBuffer(bufferMatrixA, offset: 0, index: 0)
        encoder.setBuffer(bufferResult, offset: 0, index: 1)
        encoder.setBytes([Int32(matrixA.width)], length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes([Int32(matrixA.height)], length: MemoryLayout<Int32>.size, index: 3)

        let threadsPerThreadgroupWidth = min(matrixA.width, 32)
        let threadsPerThreadgroupHeight = min(matrixA.height, 32)
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupWidth, height: threadsPerThreadgroupHeight, depth: 1)

        let threadGroupsWidth = Int(ceil(Double(matrixA.width) / Double(threadsPerThreadgroup.width)))
        let threadGroupsHeight = Int(ceil(Double(matrixA.height) / Double(threadsPerThreadgroup.height)))
        let threadGroups = MTLSize(width: threadGroupsWidth, height: threadGroupsHeight, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let output = bufferResult!.contents().bindMemory(to: Float.self, capacity: resultSize)
        let matrixResult = Matrix<Float>(elements: Array(UnsafeBufferPointer(start: output, count: resultSize)), height: matrixA.width)
        return matrixResult
    }
}