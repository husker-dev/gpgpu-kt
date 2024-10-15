
package com.huskerdev.gpkt.apis.metal


expect class MTLDevice
expect class MTLCommandQueue
expect class MTLCommandBuffer
expect class MTLLibrary
expect class MTLFunction
expect class MTLComputePipelineState
expect class MTLComputeCommandEncoder
expect class MTLBuffer

internal expect fun mtlCopyAllDevices(): Array<MTLDevice>

internal expect fun mtlGetDeviceName(device: MTLDevice): String
internal expect fun mtlNewCommandQueue(device: MTLDevice): MTLCommandQueue
internal expect fun mtlNewCommandBuffer(queue: MTLCommandQueue): MTLCommandBuffer

internal expect fun mtlCreateLibrary(device: MTLDevice, source: String): MTLLibrary
internal expect fun mtlGetFunction(library: MTLLibrary, name: String): MTLFunction
internal expect fun mtlCreatePipeline(device: MTLDevice, function: MTLFunction): MTLComputePipelineState
internal expect fun mtlCreateCommandEncoder(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState): MTLComputeCommandEncoder

internal expect fun mtlDeallocBuffer(buffer: MTLBuffer)
internal expect fun mtlDeallocLibrary(library: MTLLibrary)
internal expect fun mtlDeallocFunction(function: MTLFunction)
internal expect fun mtlDeallocCommandQueue(queue: MTLCommandQueue)
internal expect fun mtlDeallocCommandBuffer(buffer: MTLCommandBuffer)
internal expect fun mtlDeallocPipeline(pipeline: MTLComputePipelineState)
internal expect fun mtlDeallocCommandEncoder(commandEncoder: MTLComputeCommandEncoder)

internal expect fun mtlCreateBuffer(device: MTLDevice, length: Int): MTLBuffer
internal expect fun mtlWrapFloats(device: MTLDevice, array: FloatArray): MTLBuffer
internal expect fun mtlWrapInts(device: MTLDevice, array: IntArray): MTLBuffer
internal expect fun mtlWrapBytes(device: MTLDevice, array: ByteArray): MTLBuffer

internal expect fun mtlReadFloats(buffer: MTLBuffer, length: Int, offset: Int): FloatArray
internal expect fun mtlReadInts(buffer: MTLBuffer, length: Int, offset: Int): IntArray
internal expect fun mtlReadBytes(buffer: MTLBuffer, length: Int, offset: Int): ByteArray

internal expect fun mtlWriteFloats(buffer: MTLBuffer, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int)
internal expect fun mtlWriteInts(buffer: MTLBuffer, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int)
internal expect fun mtlWriteBytes(buffer: MTLBuffer, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int)

internal expect fun mtlSetBufferAt(commandEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: Int)
internal expect fun mtlSetFloatAt(commandEncoder: MTLComputeCommandEncoder, value: Float, index: Int)
internal expect fun mtlSetIntAt(commandEncoder: MTLComputeCommandEncoder, value: Int, index: Int)
internal expect fun mtlSetByteAt(commandEncoder: MTLComputeCommandEncoder, value: Byte, index: Int)

internal expect fun mtlExecute(commandBuffer: MTLCommandBuffer, commandEncoder: MTLComputeCommandEncoder, instances: Int)

class FailedToAllocateMemoryException(size: Int):
        Exception("Failed to allocate memory with size: $size bytes")
