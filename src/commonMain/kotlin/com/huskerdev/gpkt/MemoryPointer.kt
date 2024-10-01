package com.huskerdev.gpkt

interface MemoryPointer<T> {
    val length: Int
    val usage: MemoryUsage

    fun dealloc()

    fun read(
        dst: T,
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    )

    fun write(
        src: T,
        length: Int = this.length,
        srcOffset: Int = 0,
        dstOffset: Int = 0
    )
}

enum class MemoryUsage {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
}

interface DoubleMemoryPointer: MemoryPointer<DoubleArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = DoubleArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface FloatMemoryPointer: MemoryPointer<FloatArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = FloatArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface LongMemoryPointer: MemoryPointer<LongArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = LongArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface IntMemoryPointer: MemoryPointer<IntArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = IntArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface ByteMemoryPointer: MemoryPointer<ByteArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = ByteArray(length).apply {
        read(this, length, 0, 0)
    }
}