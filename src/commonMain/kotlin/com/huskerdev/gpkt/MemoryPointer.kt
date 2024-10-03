package com.huskerdev.gpkt

interface MemoryPointer<T> {
    val length: Int
    val usage: MemoryUsage

    fun dealloc()
}

interface SyncMemoryPointer<T>: MemoryPointer<T> {
    fun write(
        src: T,
        length: Int = this.length,
        srcOffset: Int = 0,
        dstOffset: Int = 0
    )
    fun read(
        dst: T,
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    )
}

interface AsyncMemoryPointer<T>: MemoryPointer<T> {
    suspend fun write(
        src: T,
        length: Int = this.length,
        srcOffset: Int = 0,
        dstOffset: Int = 0
    )
    suspend fun read(
        dst: T,
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    )
}

enum class MemoryUsage {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
}

interface SyncDoubleMemoryPointer: SyncMemoryPointer<DoubleArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = DoubleArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface SyncFloatMemoryPointer: SyncMemoryPointer<FloatArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = FloatArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface SyncIntMemoryPointer: SyncMemoryPointer<IntArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = IntArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface SyncByteMemoryPointer: SyncMemoryPointer<ByteArray>{
    fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = ByteArray(length).apply {
        read(this, length, 0, 0)
    }
}

// ===================
//       Async
// ===================

interface AsyncDoubleMemoryPointer: AsyncMemoryPointer<DoubleArray>{
    suspend fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = DoubleArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface AsyncFloatMemoryPointer: AsyncMemoryPointer<FloatArray>{
    suspend fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = FloatArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface AsyncIntMemoryPointer: AsyncMemoryPointer<IntArray>{
    suspend fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = IntArray(length).apply {
        read(this, length, 0, 0)
    }
}

interface AsyncByteMemoryPointer: AsyncMemoryPointer<ByteArray>{
    suspend fun read(
        length: Int = this.length,
        dstOffset: Int = 0,
        srcOffset: Int = 0
    ) = ByteArray(length).apply {
        read(this, length, 0, 0)
    }
}