package com.huskerdev.gpkt.apis.metal

import kotlinx.cinterop.*
import platform.Foundation.NSError
import platform.Metal.*
import platform.posix.memcpy


class MTLDeviceN(val ptr: MTLDeviceProtocol): MTLDevice
class MTLCommandQueueN(val ptr: MTLCommandQueueProtocol): MTLCommandQueue
class MTLCommandBufferN(val ptr: MTLCommandBufferProtocol): MTLCommandBuffer
class MTLLibraryN(val ptr: MTLLibraryProtocol): MTLLibrary
class MTLFunctionN(val ptr: MTLFunctionProtocol): MTLFunction
class MTLComputePipelineStateN(val ptr: MTLComputePipelineStateProtocol): MTLComputePipelineState
class MTLComputeCommandEncoderN(val ptr: MTLComputeCommandEncoderProtocol): MTLComputeCommandEncoder
class MTLBufferN(val ptr: MTLBufferProtocol): MTLBuffer

val MTLDevice.ptr: MTLDeviceProtocol
    get() = (this as MTLDeviceN).ptr
val MTLCommandQueue.ptr: MTLCommandQueueProtocol
    get() = (this as MTLCommandQueueN).ptr
val MTLCommandBuffer.ptr: MTLCommandBufferProtocol
    get() = (this as MTLCommandBufferN).ptr
val MTLLibrary.ptr: MTLLibraryProtocol
    get() = (this as MTLLibraryN).ptr
val MTLFunction.ptr: MTLFunctionProtocol
    get() = (this as MTLFunctionN).ptr
val MTLComputePipelineState.ptr: MTLComputePipelineStateProtocol
    get() = (this as MTLComputePipelineStateN).ptr
val MTLComputeCommandEncoder.ptr: MTLComputeCommandEncoderProtocol
    get() = (this as MTLComputeCommandEncoderN).ptr
val MTLBuffer.ptr: MTLBufferProtocol
    get() = (this as MTLBufferN).ptr

expect fun getDevices(): Array<MTLDeviceProtocol>

@OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
actual fun createMetal(): Metal = object: Metal(){
    override fun copyAllDevices(): Array<MTLDevice> = getDevices().map {
        MTLDeviceN(it)
    }.toTypedArray()

    override fun getDeviceName(device: MTLDevice) =
        device.ptr.name

    override fun newCommandQueue(device: MTLDevice) =
        MTLCommandQueueN(device.ptr.newCommandQueue()!!)

    override fun newCommandBuffer(queue: MTLCommandQueue) =
        MTLCommandBufferN(queue.ptr.commandBuffer()!!)

    override fun createLibrary(source: String) = memScoped {
        val err = alloc<ObjCObjectVar<NSError?>>()
        MTLLibraryN(device.ptr.newLibraryWithSource(source, null, err.ptr)
            ?: throw Exception("Failed to compile Metal library:\n${err.value!!.localizedDescription}"))
    }

    override fun getFunction(library: MTLLibrary, name: String) =
        MTLFunctionN(library.ptr.newFunctionWithName(name)!!)

    override fun createPipeline(device: MTLDevice, function: MTLFunction) = memScoped {
        val err = alloc<ObjCObjectVar<NSError?>>()
        MTLComputePipelineStateN(device.ptr.newComputePipelineStateWithFunction(function.ptr, err.ptr)
            ?: throw Exception("Failed to create pipeline for Metal function:\n${err.value!!.localizedDescription}"))
    }

    override fun createCommandEncoder(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState): MTLComputeCommandEncoder {
        val encoder = commandBuffer.ptr.computeCommandEncoder()!!
        encoder.setComputePipelineState(pipeline.ptr)
        return MTLComputeCommandEncoderN(encoder)
    }

    override fun setBufferAt(commandEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: Int) {
        commandEncoder.ptr.setBuffer(buffer.ptr, 0u, index.toULong())
    }

    override fun setFloatAt(commandEncoder: MTLComputeCommandEncoder, value: Float, index: Int) = memScoped {
        val valueVar = alloc(value)
        commandEncoder.ptr.setBytes(valueVar.ptr, Float.SIZE_BYTES.toULong(), index.toULong())
    }

    override fun setIntAt(commandEncoder: MTLComputeCommandEncoder, value: Int, index: Int) = memScoped {
        val valueVar = alloc(value)
        commandEncoder.ptr.setBytes(valueVar.ptr, Int.SIZE_BYTES.toULong(), index.toULong())
    }

    override fun setByteAt(commandEncoder: MTLComputeCommandEncoder, value: Byte, index: Int) = memScoped {
        val valueVar = alloc(value)
        commandEncoder.ptr.setBytes(valueVar.ptr, 1u, index.toULong())
    }

    override fun execute(commandBuffer: MTLCommandBuffer, commandEncoder: MTLComputeCommandEncoder, instances: Int){
        val gridSize = MTLSizeMake(instances.toULong(), 1u, 1u)
        val threadGroupSize = MTLSizeMake(instances.toULong(), 1u, 1u)

        commandEncoder.ptr.dispatchThreads(gridSize, threadGroupSize)
        commandEncoder.ptr.endEncoding()
        commandBuffer.ptr.commit()
        commandBuffer.ptr.waitUntilCompleted()
    }

    override fun deallocBuffer(buffer: MTLBuffer){
        buffer.ptr.setPurgeableState(MTLPurgeableStateEmpty)
        objc_release(buffer.objcPtr())
    }

    override fun deallocLibrary(library: MTLLibrary) =
        objc_release(library.objcPtr())

    override fun deallocFunction(function: MTLFunction) =
        objc_release(function.objcPtr())

    override fun deallocCommandQueue(queue: MTLCommandQueue) {
        objc_release(queue.objcPtr())
    }

    override fun deallocCommandBuffer(buffer: MTLCommandBuffer) {
        objc_release(buffer.objcPtr())
    }

    override fun deallocPipeline(pipeline: MTLComputePipelineState) =
        objc_release(pipeline.objcPtr())

    override fun deallocCommandEncoder(commandEncoder: MTLComputeCommandEncoder) =
        objc_release(commandEncoder.objcPtr())

    override fun createBuffer(length: Int) =
        MTLBufferN(device.ptr.newBufferWithLength(length.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(length)
        )

    override fun wrapFloats(array: FloatArray) = array.usePinned {
        val size = array.size * Float.SIZE_BYTES
        MTLBufferN(device.ptr.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(size)
        )
    }

    override fun wrapInts(array: IntArray) = array.usePinned {
        val size = array.size * Int.SIZE_BYTES
        MTLBufferN(device.ptr.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(size)
        )
    }

    override fun wrapBytes(array: ByteArray) = array.usePinned {
        val size = array.size
        MTLBufferN(device.ptr.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
            ?: throw FailedToAllocateMemoryException(size)
        )
    }

    override fun readFloats(buffer: MTLBuffer, length: Int, offset: Int) = FloatArray(length).apply {
        usePinned {
            memcpy(
                it.addressOf(0),
                interpretCPointer<CPointed>(
                    buffer.ptr.contents()!!.getRawValue() + offset.toLong() * Float.SIZE_BYTES
                ),
                (length * Float.SIZE_BYTES).toULong())
        }
    }

    override fun readInts(buffer: MTLBuffer, length: Int, offset: Int) = IntArray(length).apply {
        usePinned {
            memcpy(
                it.addressOf(0),
                interpretCPointer<CPointed>(
                    buffer.ptr.contents()!!.getRawValue() + offset.toLong() * Int.SIZE_BYTES
                ),
                (length * Int.SIZE_BYTES).toULong())
        }
    }

    override fun readBytes(buffer: MTLBuffer, length: Int, offset: Int) = ByteArray(length).apply {
        usePinned {
            memcpy(
                it.addressOf(0),
                interpretCPointer<CPointed>(
                    buffer.ptr.contents()!!.getRawValue() + offset.toLong()
                ),
                length.toULong())
        }
    }

    override fun writeFloats(buffer: MTLBuffer, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int){
        src.usePinned {
            memcpy(
                interpretCPointer<CPointed>(
                    buffer.ptr.contents()!!.getRawValue() + dstOffset.toLong() * Float.SIZE_BYTES
                ),
                it.addressOf(srcOffset),
                (length * Float.SIZE_BYTES).toULong()
            )
        }
    }

    override fun writeInts(buffer: MTLBuffer, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int){
        src.usePinned {
            memcpy(
                interpretCPointer<CPointed>(
                    buffer.ptr.contents()!!.getRawValue() + dstOffset.toLong() * Int.SIZE_BYTES
                ),
                it.addressOf(srcOffset),
                (length * Int.SIZE_BYTES).toULong()
            )
        }
    }

    override fun writeBytes(buffer: MTLBuffer, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int){
        src.usePinned {
            memcpy(
                interpretCPointer<CPointed>(
                    buffer.ptr.contents()!!.getRawValue() + dstOffset.toLong()
                ),
                it.addressOf(srcOffset),
                length.toULong()
            )
        }
    }

}