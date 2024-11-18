package com.huskerdev.gpkt

interface MemoryPointer<T>: GPResource {
    val context: GPContext
    val length: Int
    val usage: MemoryUsage
}

interface SyncMemoryPointer<T>: MemoryPointer<T> {
    override val context: GPSyncContext

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
    override val context: GPAsyncContext

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

// ===================
//        Sync
// ===================

interface SyncFloatMemoryPointer: SyncMemoryPointer<FloatArray>
interface SyncIntMemoryPointer: SyncMemoryPointer<IntArray>
interface SyncByteMemoryPointer: SyncMemoryPointer<ByteArray>

// ===================
//       Async
// ===================

interface AsyncFloatMemoryPointer: AsyncMemoryPointer<FloatArray>
interface AsyncIntMemoryPointer: AsyncMemoryPointer<IntArray>
interface AsyncByteMemoryPointer: AsyncMemoryPointer<ByteArray>