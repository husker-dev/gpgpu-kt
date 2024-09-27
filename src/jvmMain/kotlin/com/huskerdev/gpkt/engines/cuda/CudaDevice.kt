package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.Scope
import jcuda.Pointer
import jcuda.Sizeof

class CudaDevice(
    requestedDeviceId: Int
): GPDevice(GPType.CUDA) {
    private val cuda = Cuda(requestedDeviceId)

    override val id = cuda.deviceId
    override val name = cuda.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray) =
        CudaFloatMemoryPointer(cuda, cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.FLOAT), array.size)

    override fun allocFloat(length: Int) =
        CudaFloatMemoryPointer(cuda, cuda.alloc(length.toLong() * Sizeof.FLOAT), length)

    override fun allocDouble(array: DoubleArray) =
        CudaDoubleMemoryPointer(cuda, cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.DOUBLE), array.size)

    override fun allocDouble(length: Int) =
        CudaDoubleMemoryPointer(cuda, cuda.alloc(length.toLong() * Sizeof.DOUBLE), length)

    override fun allocLong(array: LongArray) =
        CudaLongMemoryPointer(cuda, cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.LONG), array.size)

    override fun allocLong(length: Int) =
        CudaLongMemoryPointer(cuda, cuda.alloc(length.toLong() * Sizeof.LONG), length)

    override fun allocInt(array: IntArray) =
        CudaIntMemoryPointer(cuda, cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.INT), array.size)

    override fun allocInt(length: Int) =
        CudaIntMemoryPointer(cuda, cuda.alloc(length.toLong() * Sizeof.INT), length)

    override fun allocByte(array: ByteArray) =
        CudaByteMemoryPointer(cuda, cuda.alloc(Pointer.to(array), array.size.toLong() * Sizeof.BYTE), array.size)

    override fun allocByte(length: Int) =
        CudaByteMemoryPointer(cuda, cuda.alloc(length.toLong() * Sizeof.BYTE), length)

    override fun compile(ast: Scope) =
        CudaProgram(cuda, ast)
}