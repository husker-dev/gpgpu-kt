package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import jcuda.Pointer
import jcuda.Sizeof

class CudaSyncDevice(
    requestedDeviceId: Int
): GPSyncDevice(GPType.CUDA) {
    private val cuda = Cuda(requestedDeviceId)

    override val id = cuda.deviceId
    override val name = cuda.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.FLOAT))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.FLOAT))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CudaSyncDoubleMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.DOUBLE))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CudaSyncDoubleMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.DOUBLE))

    override fun allocLong(array: LongArray, usage: MemoryUsage) =
        CudaSyncLongMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.LONG))

    override fun allocLong(length: Int, usage: MemoryUsage) =
        CudaSyncLongMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.LONG))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.INT))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.INT))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.BYTE))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.BYTE))

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
        CudaAsyncFloatMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.FLOAT))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.FLOAT))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CudaAsyncDoubleMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.DOUBLE))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CudaAsyncDoubleMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.DOUBLE))

    override fun allocLong(array: LongArray, usage: MemoryUsage) =
        CudaAsyncLongMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.LONG))

    override fun allocLong(length: Int, usage: MemoryUsage) =
        CudaAsyncLongMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.LONG))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.INT))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.INT))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.BYTE))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.BYTE))

    override fun compile(ast: ScopeStatement) =
        CudaProgram(cuda, ast)
}