package com.huskerdev.gpkt.apis.metal

import ca.weblite.objc.NSObject
import ca.weblite.objc.Proxy
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.Structure
import com.sun.jna.ptr.PointerByReference

private const val MTLPurgeableStateEmpty = 4
private const val MTLStorageModeShared = 0
private const val MTLResourceStorageModeShift = 4
private const val MTLResourceStorageModeShared = MTLStorageModeShared shl MTLResourceStorageModeShift

actual class MTLDevice(val ptr: Proxy)
actual class MTLCommandQueue(val ptr: Proxy)
actual class MTLCommandBuffer(val ptr: Proxy)
actual class MTLLibrary(val ptr: Proxy)
actual class MTLFunction(val ptr: Proxy)
actual class MTLComputePipelineState(val ptr: Proxy)
actual class MTLComputeCommandEncoder(val ptr: Proxy)
actual class MTLBuffer(val ptr: Proxy)

class MTLSize(
    @JvmField var width: Long = 0,
    @JvmField var height: Long = 0,
    @JvmField var depth: Long = 0,
): Structure(), Structure.ByReference

interface MTL: Library {
    companion object {
        val INSTANCE = Native.load("Metal", MTL::class.java)!!
    }
    fun MTLCopyAllDevices(): Pointer
}

internal actual fun mtlCopyAllDevices(): Array<MTLDevice> =
    NSObject.load(MTL.INSTANCE.MTLCopyAllDevices()).run {
        Array(sendInt("count")){
            MTLDevice(sendProxy("objectAtIndex:", it))
        }
    }

internal actual fun mtlGetDeviceName(device: MTLDevice) =
    device.ptr.sendString("name")

internal actual fun mtlNewCommandQueue(device: MTLDevice) =
    MTLCommandQueue(device.ptr.sendProxy("newCommandQueue"))

internal actual fun mtlNewCommandBuffer(queue: MTLCommandQueue) =
    MTLCommandBuffer(queue.ptr.sendProxy("commandBuffer"))

internal actual fun mtlCreateLibrary(
    device: MTLDevice,
    source: String
) :MTLLibrary {
    val err = PointerByReference()
    return MTLLibrary(device.ptr.sendProxy("newLibraryWithSource:options:error:", source, Pointer.NULL, err)
        ?: throw Exception("Failed to compile Metal library:\n${NSObject.load(err.value).sendString("localizedDescription")}"))
}

internal actual fun mtlGetFunction(
    library: MTLLibrary,
    name: String
) = MTLFunction(library.ptr.sendProxy("newFunctionWithName:", name))

internal actual fun mtlCreatePipeline(
    device: MTLDevice,
    function: MTLFunction
) = MTLComputePipelineState(
    device.ptr.sendProxy("newComputePipelineStateWithFunction:", function.ptr, "error:", null)
)

internal actual fun mtlCreateCommandEncoder(
    commandBuffer: MTLCommandBuffer,
    pipeline: MTLComputePipelineState
) = MTLComputeCommandEncoder(commandBuffer.ptr.sendProxy("computeCommandEncoder")).apply {
    ptr.send("setComputePipelineState", pipeline.ptr)
}

internal actual fun mtlDeallocBuffer(buffer: MTLBuffer) {
    buffer.ptr.send("setPurgeableState:", MTLPurgeableStateEmpty)
    buffer.ptr.send("release")
}

internal actual fun mtlDeallocLibrary(library: MTLLibrary) =
    library.ptr.send("release") as Unit

internal actual fun mtlDeallocFunction(function: MTLFunction) =
    function.ptr.send("release") as Unit

internal actual fun mtlDeallocCommandQueue(queue: MTLCommandQueue) =
    queue.ptr.send("release") as Unit

internal actual fun mtlDeallocCommandBuffer(buffer: MTLCommandBuffer) =
    buffer.ptr.send("release") as Unit

internal actual fun mtlDeallocPipeline(pipeline: MTLComputePipelineState) =
    pipeline.ptr.send("release") as Unit

internal actual fun mtlDeallocCommandEncoder(commandEncoder: MTLComputeCommandEncoder) =
    commandEncoder.ptr.send("release") as Unit


internal actual fun mtlCreateBuffer(device: MTLDevice, length: Int) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithLength:options:", length, MTLResourceStorageModeShared))

internal actual fun mtlWrapFloats(device: MTLDevice, array: FloatArray) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithBytes:length:options:", array, array.size * Float.SIZE_BYTES, MTLResourceStorageModeShared))

internal actual fun mtlWrapInts(device: MTLDevice, array: IntArray) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithBytes:length:options:", array, array.size * Int.SIZE_BYTES, MTLResourceStorageModeShared))

internal actual fun mtlWrapBytes(device: MTLDevice, array: ByteArray) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithBytes:length:options:", array, array.size, MTLResourceStorageModeShared))


internal actual fun mtlReadFloats(buffer: MTLBuffer, length: Int, offset: Int) =
    buffer.ptr.sendPointer("contents").getFloatArray(0, length)

internal actual fun mtlReadInts(buffer: MTLBuffer, length: Int, offset: Int) =
    buffer.ptr.sendPointer("contents").getIntArray(0, length)

internal actual fun mtlReadBytes(buffer: MTLBuffer, length: Int, offset: Int) =
    buffer.ptr.sendPointer("contents").getByteArray(0, length)


internal actual fun mtlWriteFloats(buffer: MTLBuffer, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int) =
    buffer.ptr.sendPointer("contents").write(dstOffset.toLong() * Float.SIZE_BYTES, src, srcOffset, length)


internal actual fun mtlWriteInts(buffer: MTLBuffer, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int) =
    buffer.ptr.sendPointer("contents").write(dstOffset.toLong() * Int.SIZE_BYTES, src, srcOffset, length)

internal actual fun mtlWriteBytes(buffer: MTLBuffer, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int) =
    buffer.ptr.sendPointer("contents").write(dstOffset.toLong(), src, srcOffset, length)

internal actual fun mtlSetBufferAt(commandEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: Int) =
    commandEncoder.ptr.send("setBuffer:offset:atIndex:", buffer, 0u, index) as Unit


internal actual fun mtlSetFloatAt(commandEncoder: MTLComputeCommandEncoder, value: Float, index: Int) =
    commandEncoder.ptr.send("setBytes:length:atIndex:", floatArrayOf(value), Float.SIZE_BYTES, index) as Unit

internal actual fun mtlSetIntAt(commandEncoder: MTLComputeCommandEncoder, value: Int, index: Int) =
    commandEncoder.ptr.send("setBytes:length:atIndex:", intArrayOf(value), Int.SIZE_BYTES, index) as Unit

internal actual fun mtlSetByteAt(commandEncoder: MTLComputeCommandEncoder, value: Byte, index: Int) =
    commandEncoder.ptr.send("setBytes:length:atIndex:", byteArrayOf(value), 1, index) as Unit


internal actual fun mtlExecute(
    commandBuffer: MTLCommandBuffer,
    commandEncoder: MTLComputeCommandEncoder,
    instances: Int
) {
    val gridSize = MTLSize(instances.toLong(), 1, 1)
    val threadGroupSize = MTLSize(instances.toLong(), 1, 1)

    commandEncoder.ptr.send("dispatchThreads:threadsPerThreadgroup:", gridSize, threadGroupSize)
    commandEncoder.ptr.send("endEncoding")
    commandBuffer.ptr.send("commit")
    commandBuffer.ptr.send("waitUntilCompleted")
}