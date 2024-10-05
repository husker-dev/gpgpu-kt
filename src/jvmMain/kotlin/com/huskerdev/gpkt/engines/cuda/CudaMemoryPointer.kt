package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import jcuda.driver.CUdeviceptr

private typealias CUDAReader<T> = (ptr: CUdeviceptr, length: Int, offset: Int) -> T
private typealias CUDAWriter<T> = (ptr: CUdeviceptr, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit

abstract class CudaMemoryPointer<T>: MemoryPointer<T> {
    abstract val cuda: Cuda
    abstract val ptr: CUdeviceptr

    override fun dealloc() =
        cuda.dealloc(ptr)

    abstract class Sync<T>(
        val reader: CUDAReader<T>,
        val writer: CUDAWriter<T>,
    ): CudaMemoryPointer<T>(), SyncMemoryPointer<T>{
        override fun read(length: Int, offset: Int) =
            reader(ptr, length, offset)
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(ptr, src, length, srcOffset, dstOffset)
    }

    abstract class Async<T>(
        val reader: CUDAReader<T>,
        val writer: CUDAWriter<T>,
    ): CudaMemoryPointer<T>(), AsyncMemoryPointer<T>{
        override suspend fun read(length: Int, offset: Int) =
            reader(ptr, length, offset)
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(ptr, src, length, srcOffset, dstOffset)
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
    cuda::readFloats, cuda::writeFloats
), SyncFloatMemoryPointer

class CudaSyncDoubleMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<DoubleArray>(
    cuda::readDoubles, cuda::writeDoubles
), SyncDoubleMemoryPointer

class CudaSyncIntMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<IntArray>(
    cuda::readInts, cuda::writeInts
), SyncIntMemoryPointer

class CudaSyncByteMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Sync<ByteArray>(
    cuda::readBytes, cuda::writeBytes
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
    cuda::readFloats, cuda::writeFloats
), AsyncFloatMemoryPointer

class CudaAsyncDoubleMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<DoubleArray>(
    cuda::readDoubles, cuda::writeDoubles
), AsyncDoubleMemoryPointer

class CudaAsyncIntMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<IntArray>(
    cuda::readInts, cuda::writeInts
), AsyncIntMemoryPointer

class CudaAsyncByteMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr
): CudaMemoryPointer.Async<ByteArray>(
    cuda::readBytes, cuda::writeBytes
), AsyncByteMemoryPointer