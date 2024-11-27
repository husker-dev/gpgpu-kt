package com.huskerdev.gpkt.apis.metal

import kotlinx.cinterop.*
import platform.Foundation.NSError
import platform.Metal.*
import platform.posix.memcpy

actual val metalSupported = true

actual class MTLDevice(val ptr: MTLDeviceProtocol)
actual class MTLCommandQueue(val ptr: MTLCommandQueueProtocol)
actual class MTLCommandBuffer(val ptr: MTLCommandBufferProtocol)
actual class MTLLibrary(val ptr: MTLLibraryProtocol)
actual class MTLFunction(val ptr: MTLFunctionProtocol)
actual class MTLComputePipelineState(val ptr: MTLComputePipelineStateProtocol)
actual class MTLComputeCommandEncoder(val ptr: MTLComputeCommandEncoderProtocol)
actual class MTLBuffer(val ptr: MTLBufferProtocol)

expect fun getDevices(): Array<MTLDeviceProtocol>

internal actual fun mtlCopyAllDevices(): Array<MTLDevice> = getDevices().map {
    MTLDevice(it)
}.toTypedArray()

internal actual fun mtlGetDeviceName(device: MTLDevice) =
    device.ptr.name

internal actual fun mtlNewCommandQueue(device: MTLDevice) =
    MTLCommandQueue(device.ptr.newCommandQueue()!!)

internal actual fun mtlNewCommandBuffer(queue: MTLCommandQueue) =
    MTLCommandBuffer(queue.ptr.commandBuffer()!!)

@OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
internal actual fun mtlCreateLibrary(device: MTLDevice, source: String) = memScoped {
    val err = alloc<ObjCObjectVar<NSError?>>()
    MTLLibrary(device.ptr.newLibraryWithSource(source, null, err.ptr)
        ?: throw Exception("Failed to compile Metal library:\n${err.value!!.localizedDescription}"))
}

internal actual fun mtlGetFunction(library: MTLLibrary, name: String) =
    MTLFunction(library.ptr.newFunctionWithName(name)!!)

@OptIn(ExperimentalForeignApi::class, BetaInteropApi::class)
internal actual fun mtlCreatePipeline(device: MTLDevice, function: MTLFunction) = memScoped {
    val err = alloc<ObjCObjectVar<NSError?>>()
    MTLComputePipelineState(device.ptr.newComputePipelineStateWithFunction(function.ptr, err.ptr)
        ?: throw Exception("Failed to create pipeline for Metal function:\n${err.value!!.localizedDescription}"))
}

internal actual fun mtlCreateCommandEncoder(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState): MTLComputeCommandEncoder {
    val encoder = commandBuffer.ptr.computeCommandEncoder()!!
    encoder.setComputePipelineState(pipeline.ptr)
    return MTLComputeCommandEncoder(encoder)
}

internal actual fun mtlSetBufferAt(commandEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: Int) {
    commandEncoder.ptr.setBuffer(buffer.ptr, 0u, index.toULong())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlSetFloatAt(commandEncoder: MTLComputeCommandEncoder, value: Float, index: Int) = memScoped {
    val valueVar = alloc(value)
    commandEncoder.ptr.setBytes(valueVar.ptr, Float.SIZE_BYTES.toULong(), index.toULong())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlSetIntAt(commandEncoder: MTLComputeCommandEncoder, value: Int, index: Int) = memScoped {
    val valueVar = alloc(value)
    commandEncoder.ptr.setBytes(valueVar.ptr, Int.SIZE_BYTES.toULong(), index.toULong())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlSetByteAt(commandEncoder: MTLComputeCommandEncoder, value: Byte, index: Int) = memScoped {
    val valueVar = alloc(value)
    commandEncoder.ptr.setBytes(valueVar.ptr, 1u, index.toULong())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlExecute(
    commandBuffer: MTLCommandBuffer,
    commandEncoder: MTLComputeCommandEncoder,
    gridSize: Int,
    threadGroupSize: Int
){
    commandEncoder.ptr.dispatchThreads(
        MTLSizeMake(gridSize.toULong(), 1u, 1u),
        MTLSizeMake(threadGroupSize.toULong(), 1u, 1u)
    )
    commandEncoder.ptr.endEncoding()
    commandBuffer.ptr.commit()
    commandBuffer.ptr.waitUntilCompleted()
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlDeallocBuffer(buffer: MTLBuffer){
    buffer.ptr.setPurgeableState(MTLPurgeableStateEmpty)
    objc_release(buffer.objcPtr())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlDeallocLibrary(library: MTLLibrary) =
    objc_release(library.objcPtr())

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlDeallocFunction(function: MTLFunction) =
    objc_release(function.objcPtr())

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlDeallocCommandQueue(queue: MTLCommandQueue) {
    objc_release(queue.objcPtr())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlDeallocCommandBuffer(buffer: MTLCommandBuffer) {
    objc_release(buffer.objcPtr())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlDeallocPipeline(pipeline: MTLComputePipelineState) =
    objc_release(pipeline.objcPtr())

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlDeallocCommandEncoder(commandEncoder: MTLComputeCommandEncoder) =
    objc_release(commandEncoder.objcPtr())

internal actual fun mtlCreateBuffer(device: MTLDevice, length: Int) =
    MTLBuffer(device.ptr.newBufferWithLength(length.toULong(), MTLResourceStorageModeShared)
        ?: throw FailedToAllocateMemoryException(length)
    )

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlWrapFloats(device: MTLDevice, array: FloatArray) = array.usePinned {
    val size = array.size * Float.SIZE_BYTES
    MTLBuffer(device.ptr.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
        ?: throw FailedToAllocateMemoryException(size)
    )
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlWrapInts(device: MTLDevice, array: IntArray) = array.usePinned {
    val size = array.size * Int.SIZE_BYTES
    MTLBuffer(device.ptr.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
        ?: throw FailedToAllocateMemoryException(size)
    )
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlWrapBytes(device: MTLDevice, array: ByteArray) = array.usePinned {
    val size = array.size
    MTLBuffer(device.ptr.newBufferWithBytes(it.addressOf(0), size.toULong(), MTLResourceStorageModeShared)
        ?: throw FailedToAllocateMemoryException(size)
    )
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlReadFloats(buffer: MTLBuffer, length: Int, offset: Int) = FloatArray(length).apply {
    usePinned {
        memcpy(
            it.addressOf(0),
            interpretCPointer<CPointed>(
                buffer.ptr.contents()!!.getRawValue() + offset.toLong() * Float.SIZE_BYTES
            ),
            (length * Float.SIZE_BYTES).toULong())
    }
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlReadInts(buffer: MTLBuffer, length: Int, offset: Int) = IntArray(length).apply {
    usePinned {
        memcpy(
            it.addressOf(0),
            interpretCPointer<CPointed>(
                buffer.ptr.contents()!!.getRawValue() + offset.toLong() * Int.SIZE_BYTES
            ),
            (length * Int.SIZE_BYTES).toULong())
    }
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlReadBytes(buffer: MTLBuffer, length: Int, offset: Int) = ByteArray(length).apply {
    usePinned {
        memcpy(
            it.addressOf(0),
            interpretCPointer<CPointed>(
                buffer.ptr.contents()!!.getRawValue() + offset.toLong()
            ),
            length.toULong())
    }
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlWriteFloats(buffer: MTLBuffer, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int){
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

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlWriteInts(buffer: MTLBuffer, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int){
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

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlWriteBytes(buffer: MTLBuffer, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int){
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

internal actual fun maxTotalThreadsPerThreadgroup(pipeline: MTLComputePipelineState) =
    pipeline.ptr.maxTotalThreadsPerThreadgroup.toInt()