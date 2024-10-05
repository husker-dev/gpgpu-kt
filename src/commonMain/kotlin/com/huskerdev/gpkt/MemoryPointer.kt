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
        length: Int = this.length,
        offset: Int = 0,
    ): T
}

interface AsyncMemoryPointer<T>: MemoryPointer<T> {
    suspend fun write(
        src: T,
        length: Int = this.length,
        srcOffset: Int = 0,
        dstOffset: Int = 0
    )
    suspend fun read(
        length: Int = this.length,
        offset: Int = 0,
    ): T
}

enum class MemoryUsage {
    READ_ONLY,
    WRITE_ONLY,
    READ_WRITE
}

interface SyncDoubleMemoryPointer: SyncMemoryPointer<DoubleArray>
interface SyncFloatMemoryPointer: SyncMemoryPointer<FloatArray>
interface SyncIntMemoryPointer: SyncMemoryPointer<IntArray>
interface SyncByteMemoryPointer: SyncMemoryPointer<ByteArray>

// ===================
//       Async
// ===================

interface AsyncDoubleMemoryPointer: AsyncMemoryPointer<DoubleArray>
interface AsyncFloatMemoryPointer: AsyncMemoryPointer<FloatArray>
interface AsyncIntMemoryPointer: AsyncMemoryPointer<IntArray>
interface AsyncByteMemoryPointer: AsyncMemoryPointer<ByteArray>