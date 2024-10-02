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

    protected fun writeImpl(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        cuda.write(ptr, wrapper(src),
            size = length.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize
        )
    }

    protected fun readImpl(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        cuda.read(ptr, wrapper(dst),
            size = length.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize
        )
    }

    abstract class Sync<T>(
        typeSize: Int,
        wrapper: (T) -> Pointer
    ): CudaMemoryPointer<T>(typeSize, wrapper), SyncMemoryPointer<T>{
        override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }

    abstract class Async<T>(
        typeSize: Int,
        wrapper: (T) -> Pointer
    ): CudaMemoryPointer<T>(typeSize, wrapper), AsyncMemoryPointer<T>{
        override suspend fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }
}

// ===================
//        Sync
// ===================

class CudaSyncFloatMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<FloatArray>(
    Sizeof.FLOAT, Pointer::to
), SyncFloatMemoryPointer

class CudaSyncDoubleMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<DoubleArray>(
    Sizeof.DOUBLE, Pointer::to
), SyncDoubleMemoryPointer

class CudaSyncLongMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<LongArray>(
    Sizeof.LONG, Pointer::to
), SyncLongMemoryPointer

class CudaSyncIntMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<IntArray>(
    Sizeof.INT, Pointer::to
), SyncIntMemoryPointer

class CudaSyncByteMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<ByteArray>(
    Sizeof.BYTE, Pointer::to
), SyncByteMemoryPointer


// ===================
//       Async
// ===================

class CudaAsyncFloatMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<FloatArray>(
    Sizeof.FLOAT, Pointer::to
), AsyncFloatMemoryPointer

class CudaAsyncDoubleMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<DoubleArray>(
    Sizeof.DOUBLE, Pointer::to
), AsyncDoubleMemoryPointer

class CudaAsyncLongMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<LongArray>(
    Sizeof.LONG, Pointer::to
), AsyncLongMemoryPointer

class CudaAsyncIntMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<IntArray>(
    Sizeof.INT, Pointer::to
), AsyncIntMemoryPointer

class CudaAsyncByteMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<ByteArray>(
    Sizeof.BYTE, Pointer::to
), AsyncByteMemoryPointer