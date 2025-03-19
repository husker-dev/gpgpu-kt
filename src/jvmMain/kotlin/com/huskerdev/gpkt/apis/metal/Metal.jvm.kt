package com.huskerdev.gpkt.apis.metal

import ca.weblite.objc.NSObject
import ca.weblite.objc.Proxy
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Pointer
import com.sun.jna.Structure
import com.sun.jna.ptr.PointerByReference

actual val metalSupported: Boolean = System.getProperty("os.name").lowercase().contains("mac")

private const val MTLStorageModeShared = 0
private const val MTLResourceStorageModeShift = 4
private const val MTLResourceStorageModeShared = MTLStorageModeShared shl MTLResourceStorageModeShift

actual abstract class ObjCDisposable {
    abstract val ptr: Proxy
}

actual class MTLDevice(override val ptr: Proxy): ObjCDisposable()
actual class MTLCommandQueue(override val ptr: Proxy): ObjCDisposable()
actual class MTLCommandBuffer(override val ptr: Proxy): ObjCDisposable()
actual class MTLLibrary(override val ptr: Proxy): ObjCDisposable()
actual class MTLFunction(override val ptr: Proxy): ObjCDisposable()
actual class MTLComputePipelineState(override val ptr: Proxy): ObjCDisposable()
actual class MTLComputeCommandEncoder(override val ptr: Proxy): ObjCDisposable()
actual class MTLBuffer(override val ptr: Proxy): ObjCDisposable()
actual class MTLArgumentEncoder(override val ptr: Proxy): ObjCDisposable()

@Suppress("unused")
class MTLSize(
    @JvmField var width: Long = 0,
    @JvmField var height: Long = 0,
    @JvmField var depth: Long = 0,
): Structure(), Structure.ByReference{
    override fun getFieldOrder() = listOf("width", "height", "depth")
}

@Suppress("unused")
class NSRange(
    @JvmField var loc: Long = 0,
    @JvmField var len: Long = 0
): Structure(), Structure.ByReference{
    override fun getFieldOrder() = listOf("loc", "len")
}

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

internal actual fun mtlGetDeviceMemory(device: MTLDevice) =
    (device.ptr.send("recommendedMaxWorkingSetSize") as Long).toULong()

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
    device.ptr.sendProxy("newComputePipelineStateWithFunction:error:", function.ptr, null)
)

internal actual fun mtlCreateCommandEncoder(
    commandBuffer: MTLCommandBuffer,
    pipeline: MTLComputePipelineState
) = MTLComputeCommandEncoder(commandBuffer.ptr.sendProxy("computeCommandEncoder")).apply {
    ptr.send("setComputePipelineState:", pipeline.ptr)
}

internal actual fun mtlCreateArgumentEncoderWithIndex(function: MTLFunction, index: Int) =
    MTLArgumentEncoder(function.ptr.sendProxy("newArgumentEncoderWithBufferIndex:", index))

internal actual fun mtlCreateAndBindArgumentBuffer(device: MTLDevice, argumentEncoder: MTLArgumentEncoder): MTLBuffer {
    val length = argumentEncoder.ptr.sendInt("encodedLength")
    val argumentBuffer = device.ptr.sendProxy("newBufferWithLength:options:", length, MTLResourceStorageModeShared)
    argumentEncoder.ptr.send("setArgumentBuffer:offset:", argumentBuffer, 0)
    return MTLBuffer(argumentBuffer)
}

internal actual fun mtlRelease(disposable: ObjCDisposable) {
    disposable.ptr.send("release")
}


internal actual fun mtlCreateBuffer(device: MTLDevice, length: Int) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithLength:options:", length, MTLResourceStorageModeShared))

internal actual fun mtlWrapFloats(device: MTLDevice, array: FloatArray) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithBytes:length:options:", array, array.size * Float.SIZE_BYTES, MTLResourceStorageModeShared))

internal actual fun mtlWrapInts(device: MTLDevice, array: IntArray) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithBytes:length:options:", array, array.size * Int.SIZE_BYTES, MTLResourceStorageModeShared))

internal actual fun mtlWrapBytes(device: MTLDevice, array: ByteArray) =
    MTLBuffer(device.ptr.sendProxy("newBufferWithBytes:length:options:", array, array.size, MTLResourceStorageModeShared))


internal actual fun mtlReadFloats(buffer: MTLBuffer, length: Int, offset: Int) =
    buffer.ptr.sendPointer("contents").getFloatArray(offset.toLong(), length)

internal actual fun mtlReadInts(buffer: MTLBuffer, length: Int, offset: Int) =
    buffer.ptr.sendPointer("contents").getIntArray(offset.toLong(), length)

internal actual fun mtlReadBytes(buffer: MTLBuffer, length: Int, offset: Int) =
    buffer.ptr.sendPointer("contents").getByteArray(offset.toLong(), length)

internal actual fun mtlWriteFloats(buffer: MTLBuffer, src: FloatArray, length: Int, srcOffset: Int, dstOffset: Int) =
    buffer.ptr.sendPointer("contents").write(dstOffset.toLong() * Float.SIZE_BYTES, src, srcOffset, length)


internal actual fun mtlWriteInts(buffer: MTLBuffer, src: IntArray, length: Int, srcOffset: Int, dstOffset: Int) =
    buffer.ptr.sendPointer("contents").write(dstOffset.toLong() * Int.SIZE_BYTES, src, srcOffset, length)

internal actual fun mtlWriteBytes(buffer: MTLBuffer, src: ByteArray, length: Int, srcOffset: Int, dstOffset: Int) =
    buffer.ptr.sendPointer("contents").write(dstOffset.toLong(), src, srcOffset, length)

internal actual fun mtlSetBufferAt(commandEncoder: MTLComputeCommandEncoder, buffer: MTLBuffer, index: Int) {
    commandEncoder.ptr.send("setBuffer:offset:atIndex:", buffer.ptr, 0, index)
}

internal actual fun mtlSetBufferAt(argumentEncoder: MTLArgumentEncoder, buffer: MTLBuffer, index: Int) {
    argumentEncoder.ptr.send("setBuffer:offset:atIndex:", buffer.ptr, 0, index)
}

internal actual fun mtlSetFloatAt(argumentEncoder: MTLArgumentEncoder, value: Float, index: Int) {
    val ptr = argumentEncoder.ptr.sendPointer("constantDataAtIndex:", index)
    ptr.setFloat(0, value)
}

internal actual fun mtlSetIntAt(argumentEncoder: MTLArgumentEncoder, value: Int, index: Int) {
    val ptr = argumentEncoder.ptr.sendPointer("constantDataAtIndex:", index)
    ptr.setInt(0, value)
}

internal actual fun mtlSetByteAt(argumentEncoder: MTLArgumentEncoder, value: Byte, index: Int) {
    val ptr = argumentEncoder.ptr.sendPointer("constantDataAtIndex:", index)
    ptr.setByte(0, value)
}

internal actual fun maxTotalThreadsPerThreadgroup(pipeline: MTLComputePipelineState) =
    pipeline.ptr.sendInt("maxTotalThreadsPerThreadgroup")

internal actual fun mtlExecute(
    commandBuffer: MTLCommandBuffer,
    commandEncoder: MTLComputeCommandEncoder,
    gridSize: Int,
    threadGroupSize: Int
) {
    commandEncoder.ptr.send(
        "dispatchThreads:threadsPerThreadgroup:",
        MTLSize(gridSize.toLong(), 1, 1),
        MTLSize(threadGroupSize.toLong(), 1, 1)
    )
    commandEncoder.ptr.send("endEncoding")
    commandBuffer.ptr.send("commit")
    commandBuffer.ptr.send("waitUntilCompleted")
}