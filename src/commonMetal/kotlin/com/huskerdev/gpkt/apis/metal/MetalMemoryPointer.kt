package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*


typealias MetalReader<T> = (buffer: MTLBuffer, length: Int, offset: Int) -> T
typealias MetalWriter<T> = (buffer: MTLBuffer, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit

abstract class MetalMemoryPointer<T>: MemoryPointer<T> {
    abstract val buffer: MTLBuffer
    override var disposed = false
        get() = field || context.disposed

    override fun dealloc() {
        if(disposed) return
        disposed = true
        mtlDeallocBuffer(buffer)
    }

    abstract class Sync<T>(
        val reader: MetalReader<T>,
        val writer: MetalWriter<T>
    ): MetalMemoryPointer<T>(), SyncMemoryPointer<T>{
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(buffer, src, length, srcOffset, dstOffset)

        override fun read(length: Int, offset: Int) =
            reader(buffer, length, offset)
    }

    abstract class Async<T>(
        val reader: MetalReader<T>,
        val writer: MetalWriter<T>
    ): MetalMemoryPointer<T>(), AsyncMemoryPointer<T> {
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writer(buffer, src, length, srcOffset, dstOffset)

        override suspend fun read(length: Int, offset: Int) =
            reader(buffer, length, offset)
    }
}

// ===================
//        Sync
// ===================

class MetalSyncFloatMemoryPointer(
    override val context: MetalSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer
): MetalMemoryPointer.Sync<FloatArray>(
    ::mtlReadFloats,
    ::mtlWriteFloats
), SyncFloatMemoryPointer

class MetalSyncIntMemoryPointer(
    override val context: MetalSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer
): MetalMemoryPointer.Sync<IntArray>(
    ::mtlReadInts,
    ::mtlWriteInts
), SyncIntMemoryPointer

class MetalSyncByteMemoryPointer(
    override val context: MetalSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer
): MetalMemoryPointer.Sync<ByteArray>(
    ::mtlReadBytes,
    ::mtlWriteBytes
), SyncByteMemoryPointer


// ===================
//       Async
// ===================

class MetalAsyncFloatMemoryPointer(
    override val context: MetalAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
): MetalMemoryPointer.Async<FloatArray>(
    ::mtlReadFloats,
    ::mtlWriteFloats
), AsyncFloatMemoryPointer

class MetalAsyncIntMemoryPointer(
    override val context: MetalAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
): MetalMemoryPointer.Async<IntArray>(
    ::mtlReadInts,
    ::mtlWriteInts
), AsyncIntMemoryPointer

class MetalAsyncByteMemoryPointer(
    override val context: MetalAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
): MetalMemoryPointer.Async<ByteArray>(
    ::mtlReadBytes,
    ::mtlWriteBytes
), AsyncByteMemoryPointer