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

    override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        copyInto(src, array!!, dstOffset, srcOffset, srcOffset + length)
    }

    override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        copyInto(array!!, dst, dstOffset, srcOffset, srcOffset + length)
    }
}

class CPUFloatMemoryPointer(
    override var array: FloatArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer<FloatArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), FloatMemoryPointer

class CPUDoubleMemoryPointer(
    override var array: DoubleArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer<DoubleArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), DoubleMemoryPointer

class CPULongMemoryPointer(
    override var array: LongArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer<LongArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), LongMemoryPointer

class CPUIntMemoryPointer(
    override var array: IntArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer<IntArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), IntMemoryPointer

class CPUByteMemoryPointer(
    override var array: ByteArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer<ByteArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) }
), ByteMemoryPointer