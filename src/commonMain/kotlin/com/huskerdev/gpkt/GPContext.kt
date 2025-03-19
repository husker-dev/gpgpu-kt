package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.objects.GPScope


interface GPContext: GPResource{
    val device: GPDevice
    val modules: GPModules

    val allocated: List<GPResource>
    val memory: Long

    fun compile(ast: GPScope): GPProgram
    fun compile(code: String) =
        compile(GPAst.parse(code, this))

    fun releaseMemory(memory: MemoryPointer<*>)
    fun releaseProgram(program: GPProgram)
}

interface GPSyncContext: GPContext {
    fun wrapFloats(array: FloatArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncFloatMemoryPointer
    fun allocFloats(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncFloatMemoryPointer

    fun wrapInts(array: IntArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncIntMemoryPointer
    fun allocInts(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncIntMemoryPointer

    fun wrapBytes(array: ByteArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncByteMemoryPointer
    fun allocBytes(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncByteMemoryPointer
}


interface GPAsyncContext: GPContext {
    fun wrapFloats(array: FloatArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncFloatMemoryPointer
    fun allocFloats(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncFloatMemoryPointer

    fun wrapInts(array: IntArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncIntMemoryPointer
    fun allocInts(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncIntMemoryPointer

    fun wrapBytes(array: ByteArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncByteMemoryPointer
    fun allocBytes(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncByteMemoryPointer
}