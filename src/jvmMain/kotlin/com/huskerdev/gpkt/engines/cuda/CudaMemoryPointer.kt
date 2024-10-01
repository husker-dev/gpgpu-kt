package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUdeviceptr


abstract class CudaMemoryPointer<T>(
    private val typeSize: Int,
    private val wrapper: (T) -> Pointer
): MemoryPointer<T> {
    abstract val cuda: Cuda
    abstract val ptr: CUdeviceptr

    override fun dealloc() = cuda.dealloc(ptr)

    override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        cuda.read(ptr, wrapper(dst),
            size = length.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize
        )
    }

    override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        cuda.write(ptr, wrapper(src),
            size = length.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize
        )
    }
}

class CudaFloatMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer<FloatArray>(
    Sizeof.FLOAT, Pointer::to
), FloatMemoryPointer

class CudaDoubleMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer<DoubleArray>(
    Sizeof.DOUBLE, Pointer::to
), DoubleMemoryPointer

class CudaLongMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer<LongArray>(
    Sizeof.LONG, Pointer::to
), LongMemoryPointer

class CudaIntMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer<IntArray>(
    Sizeof.INT, Pointer::to
), IntMemoryPointer

class CudaByteMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer<ByteArray>(
    Sizeof.BYTE, Pointer::to
), ByteMemoryPointer