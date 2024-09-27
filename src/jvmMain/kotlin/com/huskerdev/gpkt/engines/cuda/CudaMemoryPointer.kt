package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUdeviceptr


interface CudaMemoryPointer {
    val ptr: CUdeviceptr
}

class CudaFloatMemoryPointer(
    private val cuda: Cuda,
    override val ptr: CUdeviceptr,
    override val length: Int
): CudaMemoryPointer, FloatMemoryPointer {
    override fun read() = FloatArray(length).apply {
        cuda.read(ptr, length.toLong() * Sizeof.FLOAT, Pointer.to(this))
    }
    override fun dealloc() = cuda.dealloc(ptr)
}

class CudaDoubleMemoryPointer(
    private val cuda: Cuda,
    override val ptr: CUdeviceptr,
    override val length: Int
): CudaMemoryPointer, DoubleMemoryPointer {
    override fun read() = DoubleArray(length).apply {
        cuda.read(ptr, length.toLong() * Sizeof.DOUBLE, Pointer.to(this))
    }
    override fun dealloc() = cuda.dealloc(ptr)
}

class CudaLongMemoryPointer(
    private val cuda: Cuda,
    override val ptr: CUdeviceptr,
    override val length: Int
): CudaMemoryPointer, LongMemoryPointer {
    override fun read() = LongArray(length).apply {
        cuda.read(ptr, length.toLong() * Sizeof.LONG, Pointer.to(this))
    }
    override fun dealloc() = cuda.dealloc(ptr)
}

class CudaIntMemoryPointer(
    private val cuda: Cuda,
    override val ptr: CUdeviceptr,
    override val length: Int
): CudaMemoryPointer, IntMemoryPointer {
    override fun read() = IntArray(length).apply {
        cuda.read(ptr, length.toLong() * Sizeof.INT, Pointer.to(this))
    }
    override fun dealloc() = cuda.dealloc(ptr)
}

class CudaByteMemoryPointer(
    private val cuda: Cuda,
    override val ptr: CUdeviceptr,
    override val length: Int
): CudaMemoryPointer, ByteMemoryPointer {
    override fun read() = ByteArray(length).apply {
        cuda.read(ptr, length.toLong() * Sizeof.BYTE, Pointer.to(this))
    }
    override fun dealloc() = cuda.dealloc(ptr)
}