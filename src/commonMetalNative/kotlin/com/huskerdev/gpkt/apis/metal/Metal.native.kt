package com.huskerdev.gpkt.apis.metal

import kotlinx.cinterop.*
import platform.Foundation.NSError
import platform.Metal.*
import platform.posix.memcpy

actual val metalSupported = true

actual abstract class ObjCDisposable

actual class MTLDevice(val ptr: MTLDeviceProtocol): ObjCDisposable()
actual class MTLCommandQueue(val ptr: MTLCommandQueueProtocol): ObjCDisposable()
actual class MTLCommandBuffer(val ptr: MTLCommandBufferProtocol): ObjCDisposable()
actual class MTLLibrary(val ptr: MTLLibraryProtocol): ObjCDisposable()
actual class MTLFunction(val ptr: MTLFunctionProtocol): ObjCDisposable()
actual class MTLComputePipelineState(val ptr: MTLComputePipelineStateProtocol): ObjCDisposable()
actual class MTLComputeCommandEncoder(val ptr: MTLComputeCommandEncoderProtocol): ObjCDisposable()
actual class MTLBuffer(val ptr: MTLBufferProtocol): ObjCDisposable()
actual class MTLArgumentEncoder(val ptr: MTLArgumentEncoderProtocol): ObjCDisposable()

expect fun getDevices(): Array<MTLDeviceProtocol>

internal actual fun mtlCopyAllDevices(): Array<MTLDevice> = getDevices().map {
    MTLDevice(it)
}.toTypedArray()

internal actual fun mtlGetDeviceName(device: MTLDevice) =
    device.ptr.name

internal actual fun mtlGetDeviceMemory(device: MTLDevice) =
    device.ptr.recommendedMaxWorkingSetSize.toULong()

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

internal actual fun mtlCreateArgumentEncoderWithIndex(function: MTLFunction, index: Int) =
    MTLArgumentEncoder(function.ptr.newArgumentEncoderWithBufferIndex(index.toULong()))

internal actual fun mtlCreateAndBindArgumentBuffer(device: MTLDevice, argumentEncoder: MTLArgumentEncoder): MTLBuffer {
    val length = argumentEncoder.ptr.encodedLength
    val argumentBuffer = device.ptr.newBufferWithLength(length, MTLResourceStorageModeShared)
        ?: throw FailedToAllocateMemoryException(length.toInt())
    argumentEncoder.ptr.setArgumentBuffer(argumentBuffer, 0u)
    return MTLBuffer(argumentBuffer)
}

internal actual fun mtlSetBufferAt(commandEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: Int) {
    commandEncoder.ptr.setBuffer(buffer.ptr, 0u, index.toULong())
}

internal actual fun mtlSetBufferAt(argumentEncoder: MTLArgumentEncoder, buffer: MTLBuffer, index: Int) {
    argumentEncoder.ptr.setBuffer(buffer.ptr, 0u, index.toULong())
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlSetFloatAt(argumentEncoder: MTLArgumentEncoder, value: Float, index: Int) = memScoped {
    val valueVar = alloc(value)
    val ptr = argumentEncoder.ptr.constantDataAtIndex(index.toULong())
    memcpy(ptr, valueVar.ptr, 4u)
    Unit
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlSetIntAt(argumentEncoder: MTLArgumentEncoder, value: Int, index: Int) = memScoped {
    val valueVar = alloc(value)
    val ptr = argumentEncoder.ptr.constantDataAtIndex(index.toULong())
    memcpy(ptr, valueVar.ptr, 4u)
    Unit
}

@OptIn(ExperimentalForeignApi::class)
internal actual fun mtlSetByteAt(argumentEncoder: MTLArgumentEncoder, value: Byte, index: Int) = memScoped {
    val valueVar = alloc(value)
    val ptr = argumentEncoder.ptr.constantDataAtIndex(index.toULong())
    memcpy(ptr, valueVar.ptr, 1u)
    Unit
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
internal actual fun mtlRelease(disposable: ObjCDisposable){
    objc_release(disposable.objcPtr())
}

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