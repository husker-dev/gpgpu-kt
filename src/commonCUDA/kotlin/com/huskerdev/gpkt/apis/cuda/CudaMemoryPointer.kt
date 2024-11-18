package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*

private typealias CUDAReader<T> = (context: CUcontext, ptr: CUdeviceptr, length: Int, offset: Int) -> T
private typealias CUDAWriter<T> = (context: CUcontext, ptr: CUdeviceptr, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit

abstract class CudaMemoryPointer<T>: MemoryPointer<T> {
    abstract val contextPeer: CUcontext

    abstract val ptr: CUdeviceptr
    override var released = false
        get() = field || context.released

    override fun release() {
        if(released) return
        released = true
        context.releaseMemory(this)
    }

    abstract class Sync<T>(
        val reader: CUDAReader<T>,
        val writer: CUDAWriter<T>,
    ): CudaMemoryPointer<T>(), SyncMemoryPointer<T>{
        override fun read(length: Int, offset: Int): T {
            assertNotReleased()
            return reader(contextPeer, ptr, length, offset)
        }
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
            assertNotReleased()
            writer(contextPeer, ptr, src, length, srcOffset, dstOffset)
        }
    }

    abstract class Async<T>(
        val reader: CUDAReader<T>,
        val writer: CUDAWriter<T>,
    ): CudaMemoryPointer<T>(), AsyncMemoryPointer<T>{
        override suspend fun read(length: Int, offset: Int): T {
            assertNotReleased()
            return reader(contextPeer, ptr, length, offset)
        }
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
            assertNotReleased()
            writer(contextPeer, ptr, src, length, srcOffset, dstOffset)
        }
    }
}

// ===================
//        Sync
// ===================

class CudaSyncFloatMemoryPointer(
    override val context: CudaSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr,
    override val contextPeer: CUcontext = context.peer
): CudaMemoryPointer.Sync<FloatArray>(
    Cuda::readFloats, Cuda::writeFloats
), SyncFloatMemoryPointer

class CudaSyncIntMemoryPointer(
    override val context: CudaSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr,
    override val contextPeer: CUcontext = context.peer
): CudaMemoryPointer.Sync<IntArray>(
    Cuda::readInts, Cuda::writeInts
), SyncIntMemoryPointer

class CudaSyncByteMemoryPointer(
    override val context: CudaSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr,
    override val contextPeer: CUcontext = context.peer
): CudaMemoryPointer.Sync<ByteArray>(
    Cuda::readBytes, Cuda::writeBytes
), SyncByteMemoryPointer


// ===================
//       Async
// ===================

class CudaAsyncFloatMemoryPointer(
    override val context: CudaAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr,
    override val contextPeer: CUcontext = context.peer
): CudaMemoryPointer.Async<FloatArray>(
    Cuda::readFloats, Cuda::writeFloats
), AsyncFloatMemoryPointer

class CudaAsyncIntMemoryPointer(
    override val context: CudaAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr,
    override val contextPeer: CUcontext = context.peer
): CudaMemoryPointer.Async<IntArray>(
    Cuda::readInts, Cuda::writeInts
), AsyncIntMemoryPointer

class CudaAsyncByteMemoryPointer(
    override val context: CudaAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: CUdeviceptr,
    override val contextPeer: CUcontext = context.peer
): CudaMemoryPointer.Async<ByteArray>(
    Cuda::readBytes, Cuda::writeBytes
), AsyncByteMemoryPointer