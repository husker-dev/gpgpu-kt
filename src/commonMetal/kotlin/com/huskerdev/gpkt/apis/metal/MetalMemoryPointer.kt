package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*


typealias MetalReader<T> = (buffer: MTLBuffer, length: Int, offset: Int) -> T
typealias MetalWriter<T> = (buffer: MTLBuffer, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit

abstract class MetalMemoryPointer<T>: MemoryPointer<T> {
    abstract val metal: Metal
    abstract val buffer: MTLBuffer
    override var disposed = false
        get() = field || context.disposed

    override fun dealloc() {
        if(disposed) return
        disposed = true
        metal.deallocBuffer(buffer)
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
    override val buffer: MTLBuffer,
    override val metal: Metal = context.metal
): MetalMemoryPointer.Sync<FloatArray>(
    metal::readFloats,
    metal::writeFloats
), SyncFloatMemoryPointer

class MetalSyncIntMemoryPointer(
    override val context: MetalSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
    override val metal: Metal = context.metal
): MetalMemoryPointer.Sync<IntArray>(
    metal::readInts,
    metal::writeInts
), SyncIntMemoryPointer

class MetalSyncByteMemoryPointer(
    override val context: MetalSyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
    override val metal: Metal = context.metal
): MetalMemoryPointer.Sync<ByteArray>(
    metal::readBytes,
    metal::writeBytes
), SyncByteMemoryPointer


// ===================
//       Async
// ===================

class MetalAsyncFloatMemoryPointer(
    override val context: MetalAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
    override val metal: Metal = context.metal,
): MetalMemoryPointer.Async<FloatArray>(
    metal::readFloats,
    metal::writeFloats
), AsyncFloatMemoryPointer

class MetalAsyncIntMemoryPointer(
    override val context: MetalAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
    override val metal: Metal = context.metal,
): MetalMemoryPointer.Async<IntArray>(
    metal::readInts,
    metal::writeInts
), AsyncIntMemoryPointer

class MetalAsyncByteMemoryPointer(
    override val context: MetalAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBuffer,
    override val metal: Metal = context.metal,
): MetalMemoryPointer.Async<ByteArray>(
    metal::readBytes,
    metal::writeBytes
), AsyncByteMemoryPointer