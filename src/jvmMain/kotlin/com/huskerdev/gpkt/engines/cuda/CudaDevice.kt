package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

class CudaSyncDevice(
    requestedDeviceId: Int
): GPSyncDevice(GPType.CUDA) {
    private val cuda = Cuda(requestedDeviceId)

    override val id = cuda.deviceId
    override val name = cuda.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(cuda, array.size, usage, cuda.alloc(FloatBuffer.wrap(array)))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Float.SIZE_BYTES))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CudaSyncDoubleMemoryPointer(cuda, array.size, usage, cuda.alloc(DoubleBuffer.wrap(array)))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CudaSyncDoubleMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Double.SIZE_BYTES))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(cuda, array.size, usage, cuda.alloc(IntBuffer.wrap(array)))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Int.SIZE_BYTES))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(cuda, array.size, usage, cuda.alloc(ByteBuffer.wrap(array)))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Byte.SIZE_BYTES))

    override fun compile(ast: ScopeStatement) =
        CudaProgram(cuda, ast)
}

class CudaAsyncDevice(
    requestedDeviceId: Int
): GPAsyncDevice(GPType.CUDA) {
    private val cuda = Cuda(requestedDeviceId)

    override val id = cuda.deviceId
    override val name = cuda.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(cuda, array.size, usage, cuda.alloc(FloatBuffer.wrap(array)))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Float.SIZE_BYTES))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CudaAsyncDoubleMemoryPointer(cuda, array.size, usage, cuda.alloc(DoubleBuffer.wrap(array)))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CudaAsyncDoubleMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Double.SIZE_BYTES))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(cuda, array.size, usage, cuda.alloc(IntBuffer.wrap(array)))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Int.SIZE_BYTES))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(cuda, array.size, usage, cuda.alloc(ByteBuffer.wrap(array)))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(cuda, length, usage, cuda.alloc(length.toLong() * Byte.SIZE_BYTES))

    override fun compile(ast: ScopeStatement) =
        CudaProgram(cuda, ast)
}