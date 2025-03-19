
package com.huskerdev.gpkt.apis.metal

expect val metalSupported: Boolean

expect abstract class ObjCDisposable

expect class MTLDevice: ObjCDisposable
expect class MTLCommandQueue: ObjCDisposable
expect class MTLCommandBuffer: ObjCDisposable
expect class MTLLibrary: ObjCDisposable
expect class MTLFunction: ObjCDisposable
expect class MTLComputePipelineState: ObjCDisposable
expect class MTLComputeCommandEncoder: ObjCDisposable
expect class MTLBuffer: ObjCDisposable
expect class MTLArgumentEncoder: ObjCDisposable

internal expect fun mtlCopyAllDevices(): Array<MTLDevice>

internal expect fun mtlGetDeviceName(device: MTLDevice): String
internal expect fun mtlGetDeviceMemory(device: MTLDevice): ULong
internal expect fun mtlNewCommandQueue(device: MTLDevice): MTLCommandQueue
internal expect fun mtlNewCommandBuffer(queue: MTLCommandQueue): MTLCommandBuffer

internal expect fun mtlCreateLibrary(device: MTLDevice, source: String): MTLLibrary
internal expect fun mtlGetFunction(library: MTLLibrary, name: String): MTLFunction
internal expect fun mtlCreatePipeline(device: MTLDevice, function: MTLFunction): MTLComputePipelineState
internal expect fun mtlCreateCommandEncoder(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState): MTLComputeCommandEncoder
internal expect fun mtlCreateArgumentEncoderWithIndex(function: MTLFunction, index: Int): MTLArgumentEncoder
internal expect fun mtlCreateAndBindArgumentBuffer(device: MTLDevice, argumentEncoder: MTLArgumentEncoder): MTLBuffer

internal expect fun mtlRelease(disposable: ObjCDisposable)

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
internal expect fun mtlSetBufferAt(argumentEncoder: MTLArgumentEncoder, buffer: MTLBuffer, index: Int)
internal expect fun mtlSetFloatAt(argumentEncoder: MTLArgumentEncoder, value: Float, index: Int)
internal expect fun mtlSetIntAt(argumentEncoder: MTLArgumentEncoder, value: Int, index: Int)
internal expect fun mtlSetByteAt(argumentEncoder: MTLArgumentEncoder, value: Byte, index: Int)

internal expect fun maxTotalThreadsPerThreadgroup(pipeline: MTLComputePipelineState): Int

internal expect fun mtlExecute(
        commandBuffer: MTLCommandBuffer,
        commandEncoder: MTLComputeCommandEncoder,
        gridSize: Int,
        threadGroupSize: Int)

class FailedToAllocateMemoryException(size: Int):
        Exception("Failed to allocate memory with size: $size bytes")
