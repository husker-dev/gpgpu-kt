package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.*

abstract class CPUMemoryPointer<T>(
    override val length: Int,
    val copyInto: (src: T, dst: T, dstOffset: Int, startIndex: Int, endIndex: Int) -> Unit
): MemoryPointer<T>{
    abstract var array: T?

    override fun dealloc() {
        array = null
    }

    protected fun writeImpl(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
        copyInto(src, array!!, dstOffset, srcOffset, srcOffset + length)

    protected fun readImpl(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
        copyInto(array!!, dst, dstOffset, srcOffset, srcOffset + length)

    abstract class Sync<T>(
        length: Int,
        copyInto: (src: T, dst: T, dstOffset: Int, startIndex: Int, endIndex: Int) -> Unit
    ): CPUMemoryPointer<T>(length, copyInto), SyncMemoryPointer<T>{
        override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }

    abstract class Async<T>(
        length: Int,
        copyInto: (src: T, dst: T, dstOffset: Int, startIndex: Int, endIndex: Int) -> Unit
    ): CPUMemoryPointer<T>(length, copyInto), AsyncMemoryPointer<T>{
        override suspend fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }
}

// ===================
//        Sync
// ===================

class CPUSyncFloatMemoryPointer(
    override var array: FloatArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Sync<FloatArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), SyncFloatMemoryPointer

class CPUSyncDoubleMemoryPointer(
    override var array: DoubleArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Sync<DoubleArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), SyncDoubleMemoryPointer

class CPUSyncIntMemoryPointer(
    override var array: IntArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Sync<IntArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), SyncIntMemoryPointer

class CPUSyncByteMemoryPointer(
    override var array: ByteArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Sync<ByteArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), SyncByteMemoryPointer

// ===================
//       Async
// ===================

class CPUAsyncFloatMemoryPointer(
    override var array: FloatArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Async<FloatArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), AsyncFloatMemoryPointer

class CPUAsyncDoubleMemoryPointer(
    override var array: DoubleArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Async<DoubleArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), AsyncDoubleMemoryPointer

class CPUAsyncIntMemoryPointer(
    override var array: IntArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Async<IntArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), AsyncIntMemoryPointer

class CPUAsyncByteMemoryPointer(
    override var array: ByteArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Async<ByteArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), AsyncByteMemoryPointer