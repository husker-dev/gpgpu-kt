
package com.huskerdev.gpkt.apis.metal


expect fun createMetal(): Metal

interface MTLDevice
interface MTLCommandQueue
interface MTLCommandBuffer
interface MTLLibrary
interface MTLFunction
interface MTLComputePipelineState
interface MTLComputeCommandEncoder
interface MTLBuffer

abstract class Metal {
    abstract fun copyAllDevices(): Array<MTLDevice>

    abstract fun getDeviceName(device: MTLDevice): String
    abstract fun newCommandQueue(device: MTLDevice): MTLCommandQueue
    abstract fun newCommandBuffer(queue: MTLCommandQueue): MTLCommandBuffer

    abstract fun createLibrary(source: String): MTLLibrary
    abstract fun getFunction(library: MTLLibrary, name: String): MTLFunction
    abstract fun createPipeline(device: MTLDevice, function: MTLFunction): MTLComputePipelineState
    abstract fun createCommandEncoder(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState): MTLComputeCommandEncoder

    abstract fun deallocBuffer(buffer: MTLBuffer)
    abstract fun deallocLibrary(library: MTLLibrary)
    abstract fun deallocFunction(function: MTLFunction)
    abstract fun deallocCommandQueue(queue: MTLCommandQueue)
    abstract fun deallocCommandBuffer(buffer: MTLCommandBuffer)
    abstract fun deallocPipeline(pipeline: MTLComputePipelineState)
    abstract fun deallocCommandEncoder(commandEncoder: MTLComputeCommandEncoder)

    abstract fun createBuffer(length: Int): MTLBuffer
    abstract fun wrapFloats(array: FloatArray): MTLBuffer
    abstract fun wrapInts(array: IntArray): MTLBuffer
    abstract fun wrapBytes(array: ByteArray): MTLBuffer

    abstract fun readFloats(buffer: MTLBuffer, length: Int, offset: Int): FloatArray
    abstract fun readInts(buffer: MTLBuffer, length: Int, offset: Int): IntArray
    abstract fun readBytes(buffer: MTLBuffer, length: Int, offset: Int): ByteArray

    abstract fun writeFloats(buffer: MTLBuffer, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int)
    abstract fun writeInts(buffer: MTLBuffer, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int)
    abstract fun writeBytes(buffer: MTLBuffer, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int)

    abstract fun setBufferAt(commandEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: Int)
    abstract fun setFloatAt(commandEncoder: MTLComputeCommandEncoder, value: Float, index: Int)
    abstract fun setIntAt(commandEncoder: MTLComputeCommandEncoder, value: Int, index: Int)
    abstract fun setByteAt(commandEncoder: MTLComputeCommandEncoder, value: Byte, index: Int)

    abstract fun execute(commandBuffer: MTLCommandBuffer, commandEncoder: MTLComputeCommandEncoder, instances: Int)

    class FailedToAllocateMemoryException(size: Int):
            Exception("Failed to allocate memory with size: $size bytes")
}