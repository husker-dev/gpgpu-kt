package com.huskerdev.gpkt

interface MemoryPointer {
    val length: Int
    fun dealloc()
}

interface DoubleMemoryPointer: MemoryPointer {
    fun read(): DoubleArray
}

interface FloatMemoryPointer: MemoryPointer {
    fun read(): FloatArray
}

interface LongMemoryPointer: MemoryPointer {
    fun read(): LongArray
}

interface IntMemoryPointer: MemoryPointer {
    fun read(): IntArray
}

interface ByteMemoryPointer: MemoryPointer {
    fun read(): ByteArray
}