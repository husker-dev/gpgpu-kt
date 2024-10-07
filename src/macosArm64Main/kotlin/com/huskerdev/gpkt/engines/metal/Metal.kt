
package com.huskerdev.gpkt.engines.metal

import kotlinx.cinterop.*
import platform.Foundation.NSError
import platform.Metal.*
import platform.posix.memcpy
import kotlin.math.max
import kotlin.math.min

class Metal(
    requestedDeviceId: Int
) {
    val deviceId: Int
    val device: MTLDeviceProtocol
    val name: String
    private val commandQueue: MTLCommandQueueProtocol
    private val commandBuffer: MTLCommandBufferProtocol

    init {
        val devices = MTLCopyAllDevices()
        deviceId = max(0, min(requestedDeviceId, devices.size))

        device = devices[deviceId] as MTLDeviceProtocol
        name = device.name
        commandQueue = device.newCommandQueue()!!
        commandBuffer = commandQueue.commandBuffer()!!
    }

    @OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
    fun createLibrary(source: String) = memScoped {
        val err = alloc<ObjCObjectVar<NSError?>>()
        device.newLibraryWithSource(source, null, err.ptr)
            ?: throw Exception("Failed to compile Metal library:\n${err.value!!.localizedDescription}")
    }

    fun getFunction(library: MTLLibraryProtocol, name: String) =
        library.newFunctionWithName(name)

    @OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
    fun createPipeline(device: MTLDeviceProtocol, function: MTLFunctionProtocol) = memScoped {
        val err = alloc<ObjCObjectVar<NSError?>>()
        device.newComputePipelineStateWithFunction(function, err.ptr)
            ?: throw Exception("Failed to create pipeline for Metal function:\n${err.value!!.localizedDescription}")
    }

    fun createCommandEncoder(pipeline: MTLComputePipelineStateProtocol): MTLComputeCommandEncoderProtocol {
        val encoder = commandBuffer.computeCommandEncoder()!!
        encoder.setComputePipelineState(pipeline)
        return encoder
    }

    fun setBufferAt(commandEncoder: MTLComputeCommandEncoderProtocol, buffer: MTLBufferProtocol, index: Int) {
        commandEncoder.setBuffer(buffer, 0u, index.toULong())
    }

    @OptIn(ExperimentalForeignApi::class)
    fun setFloatAt(commandEncoder: MTLComputeCommandEncoderProtocol, value: Float, index: Int) = memScoped {
        val valueVar = alloc(value)
        commandEncoder.setBytes(valueVar.ptr, Float.SIZE_BYTES.toULong(), index.toULong())
    }

    @OptIn(ExperimentalForeignApi::class)
    fun setIntAt(commandEncoder: MTLComputeCommandEncoderProtocol, value: Int, index: Int) = memScoped {
        val valueVar = alloc(value)
        commandEncoder.setBytes(valueVar.ptr, Int.SIZE_BYTES.toULong(), index.toULong())
    }

    @OptIn(ExperimentalForeignApi::class)
    fun setByteAt(commandEncoder: MTLComputeCommandEncoderProtocol, value: Byte, index: Int) = memScoped {
        val valueVar = alloc(value)
        commandEncoder.setBytes(valueVar.ptr, 1u, index.toULong())
    }

    @OptIn(ExperimentalForeignApi::class)
    fun execute(commandEncoder: MTLComputeCommandEncoderProtocol, instances: Int){
        val gridSize = MTLSizeMake(instances.toULong(), 1u, 1u)
        val threadGroupSize = MTLSizeMake(instances.toULong(), 1u, 1u)

        commandEncoder.dispatchThreads(gridSize, threadGroupSize)
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    @OptIn(ExperimentalForeignApi::class)
    fun deallocBuffer(buffer: MTLBufferProtocol){
        buffer.setPurgeableState(MTLPurgeableStateEmpty)
        objc_release(buffer.objcPtr())
    }

    fun createBuffer(length: Int) =
        device.newBufferWithLength(length.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(length)

    @OptIn(ExperimentalForeignApi::class)
    fun wrapFloats(array: FloatArray) = array.usePinned {
        val size = array.size * Float.SIZE_BYTES
        device.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(size)
    }

    @OptIn(ExperimentalForeignApi::class)
    fun wrapInts(array: IntArray) = array.usePinned {
        val size = array.size * Int.SIZE_BYTES
        device.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(size)
    }

    @OptIn(ExperimentalForeignApi::class)
    fun wrapBytes(array: ByteArray) = array.usePinned {
        val size = array.size
        device.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(size)
    }

    @OptIn(ExperimentalForeignApi::class)
    fun readFloats(buffer: MTLBufferProtocol, length: Int, offset: Int) = FloatArray(length).apply {
        usePinned {
            memcpy(
                it.addressOf(0),
                interpretCPointer<CPointed>(buffer.contents()!!.getRawValue() + offset.toLong() * Float.SIZE_BYTES),
                (length * Float.SIZE_BYTES).toULong())
        }
    }

    @OptIn(ExperimentalForeignApi::class)
    fun readInts(buffer: MTLBufferProtocol, length: Int, offset: Int) = IntArray(length).apply {
        usePinned {
            memcpy(
                it.addressOf(0),
                interpretCPointer<CPointed>(buffer.contents()!!.getRawValue() + offset.toLong() * Int.SIZE_BYTES),
                (length * Int.SIZE_BYTES).toULong())
        }
    }

    @OptIn(ExperimentalForeignApi::class)
    fun readBytes(buffer: MTLBufferProtocol, length: Int, offset: Int) = ByteArray(length).apply {
        usePinned {
            memcpy(
                it.addressOf(0),
                interpretCPointer<CPointed>(buffer.contents()!!.getRawValue() + offset.toLong()),
                length.toULong())
        }
    }

    @OptIn(ExperimentalForeignApi::class)
    fun writeFloats(buffer: MTLBufferProtocol, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int){
        src.usePinned {
            memcpy(
                interpretCPointer<CPointed>(buffer.contents()!!.getRawValue() + dstOffset.toLong() * Float.SIZE_BYTES),
                it.addressOf(srcOffset),
                (length * Float.SIZE_BYTES).toULong()
            )
        }
    }

    @OptIn(ExperimentalForeignApi::class)
    fun writeInts(buffer: MTLBufferProtocol, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int){
        src.usePinned {
            memcpy(
                interpretCPointer<CPointed>(buffer.contents()!!.getRawValue() + dstOffset.toLong() * Int.SIZE_BYTES),
                it.addressOf(srcOffset),
                (length * Int.SIZE_BYTES).toULong()
            )
        }
    }

    @OptIn(ExperimentalForeignApi::class)
    fun writeBytes(buffer: MTLBufferProtocol, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int){
        src.usePinned {
            memcpy(
                interpretCPointer<CPointed>(buffer.contents()!!.getRawValue() + dstOffset.toLong()),
                it.addressOf(srcOffset),
                length.toULong()
            )
        }
    }

    class FailedToAllocateMemoryException(size: Int):
            Exception("Failed to allocate memory with size: $size bytes")

}