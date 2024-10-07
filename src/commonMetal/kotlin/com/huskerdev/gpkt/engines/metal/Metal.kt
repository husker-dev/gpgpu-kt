
package com.huskerdev.gpkt.engines.metal

import kotlin.math.max
import kotlin.math.min

expect fun createMetal(requestedDeviceId: Int): Metal

interface MTLDevice
interface MTLCommandQueue
interface MTLCommandBuffer
interface MTLLibrary
interface MTLFunction
interface MTLComputePipelineState
interface MTLComputeCommandEncoder
interface MTLBuffer

abstract class Metal(
    private val requestedDeviceId: Int
) {
    var deviceId: Int = 0
    lateinit var device: MTLDevice
    lateinit var name: String
    protected lateinit var commandQueue: MTLCommandQueue
    protected lateinit var commandBuffer: MTLCommandBuffer

    fun init(): Metal{
        val devices = copyAllDevices()
        deviceId = max(0, min(requestedDeviceId, devices.size))

        device = devices[deviceId]
        name = getDeviceName(device)
        commandQueue = newCommandQueue(device)
        commandBuffer = newCommandBuffer(commandQueue)
        return this
    }

    protected abstract fun copyAllDevices(): Array<MTLDevice>

    protected abstract fun getDeviceName(device: MTLDevice): String
    protected abstract fun newCommandQueue(device: MTLDevice): MTLCommandQueue
    protected abstract fun newCommandBuffer(queue: MTLCommandQueue): MTLCommandBuffer

    abstract fun createLibrary(source: String): MTLLibrary
    abstract fun getFunction(library: MTLLibrary, name: String): MTLFunction
    abstract fun createPipeline(device: MTLDevice, function: MTLFunction): MTLComputePipelineState
    abstract fun createCommandEncoder(pipeline: MTLComputePipelineState): MTLComputeCommandEncoder

    abstract fun deallocBuffer(buffer: MTLBuffer)
    abstract fun deallocLibrary(library: MTLLibrary)
    abstract fun deallocFunction(function: MTLFunction)
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

    abstract fun execute(commandEncoder: MTLComputeCommandEncoder, instances: Int)

    class FailedToAllocateMemoryException(size: Int):
            Exception("Failed to allocate memory with size: $size bytes")

}