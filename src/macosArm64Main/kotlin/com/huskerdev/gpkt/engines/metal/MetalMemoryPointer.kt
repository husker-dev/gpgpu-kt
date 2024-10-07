package com.huskerdev.gpkt.engines.metal

import com.huskerdev.gpkt.*
import platform.Metal.MTLBufferProtocol


typealias MetalReader<T> = (buffer: MTLBufferProtocol, length: Int, offset: Int) -> T
typealias MetalWriter<T> = (buffer: MTLBufferProtocol, src: T, length: Int, srcOffset: Int, dstOffset: Int) -> Unit

abstract class MetalMemoryPointer<T>: MemoryPointer<T> {
    abstract val metal: Metal
    abstract val buffer: MTLBufferProtocol

    override fun dealloc() =
        metal.deallocBuffer(buffer)

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
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Sync<FloatArray>(
    metal::readFloats,
    metal::writeFloats
), SyncFloatMemoryPointer

class MetalSyncDoubleMemoryPointer(
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Sync<DoubleArray>(
    metal::readDoubles,
    metal::writeDoubles
), SyncDoubleMemoryPointer

class MetalSyncIntMemoryPointer(
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Sync<IntArray>(
    metal::readInts,
    metal::writeInts
), SyncIntMemoryPointer

class MetalSyncByteMemoryPointer(
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Sync<ByteArray>(
    metal::readBytes,
    metal::writeBytes
), SyncByteMemoryPointer


// ===================
//       Async
// ===================

class MetalAsyncFloatMemoryPointer(
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Async<FloatArray>(
    metal::readFloats,
    metal::writeFloats
), AsyncFloatMemoryPointer

class MetalAsyncDoubleMemoryPointer(
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Async<DoubleArray>(
    metal::readDoubles,
    metal::writeDoubles
), AsyncDoubleMemoryPointer

class MetalAsyncIntMemoryPointer(
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Async<IntArray>(
    metal::readInts,
    metal::writeInts
), AsyncIntMemoryPointer

class MetalAsyncByteMemoryPointer(
    override val metal: Metal,
    override val length: Int,
    override val usage: MemoryUsage,
    override val buffer: MTLBufferProtocol
): MetalMemoryPointer.Async<ByteArray>(
    metal::readBytes,
    metal::writeBytes
), AsyncByteMemoryPointer