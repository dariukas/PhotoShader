import Metal
import CoreGraphics

public final class ThresholdPipeline {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState
    
    public init() throws {
        // Initialize the default hardware interface layer
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "MetalPipeline", code: 1, userInfo: [NSLocalizedDescriptionKey: "Metal not supported"])
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw NSError(domain: "MetalPipeline", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to build command queue"])
        }
        self.device = device
        self.commandQueue = commandQueue
        
        // Compile the custom shader function into an executable pipeline state
        let library = try device.makeDefaultLibrary(bundle: Bundle.main)
        guard let kernelFunction = library.makeFunction(name: "imageThreshold") else {
            throw NSError(domain: "MetalPipeline", code: 3, userInfo: [NSLocalizedDescriptionKey: "Kernel not found"])
        }
        self.pipelineState = try device.makeComputePipelineState(function: kernelFunction)
    }
    
    public func execute(inputTexture: MTLTexture, outputTexture: MTLTexture, thresholdValue: Float) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(inputTexture, index: 0)
        computeEncoder.setTexture(outputTexture, index: 1)
        
        // Pass the structural threshold variable natively directly into buffer slot 0
        var mutableThreshold = thresholdValue
        computeEncoder.setBytes(&mutableThreshold, length: MemoryLayout<Float>.size, index: 0)
        
        // M5 Tuning Strategy: Calculate execution grid allocations based on actual hardware thread Execution Width
        let executionWidth = pipelineState.threadExecutionWidth // Typically 32 or 64 on Apple Silicon
        let maxThreads = pipelineState.maxTotalThreadsPerThreadgroup
        
        // Build optimal 2D thread blocks matching execution warp configurations
        let threadgroupSize = MTLSize(width: executionWidth, height: maxThreads / executionWidth, depth: 1)
        
        // Define the total macro execution grid size to explicitly match image dimensions
        let gridSize = MTLSize(width: inputTexture.width, height: inputTexture.height, depth: 1)
        
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted() // Keeps execution synchronous for debugging profiles
    }
}
