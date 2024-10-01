package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import jcuda.Pointer
import jcuda.Sizeof

class CudaDevice(
    requestedDeviceId: Int
): GPDevice(GPType.CUDA) {
    private val cuda = Cuda(requestedDeviceId)

    override val id = cuda.deviceId
    override val name = cuda.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CudaFloatMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.FLOAT))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CudaFloatMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.FLOAT))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CudaDoubleMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.DOUBLE))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CudaDoubleMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.DOUBLE))

    override fun allocLong(array: LongArray, usage: MemoryUsage) =
        CudaLongMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.LONG))

    override fun allocLong(length: Int, usage: MemoryUsage) =
        CudaLongMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.LONG))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CudaIntMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.INT))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CudaIntMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.INT))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CudaByteMemoryPointer(cuda, array.size, usage,
            cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.BYTE))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CudaByteMemoryPointer(cuda, length, usage,
            cuda.alloc(length.toLong() * Sizeof.BYTE))

    override fun compile(ast: ScopeStatement) =
        CudaProgram(cuda, ast)
}