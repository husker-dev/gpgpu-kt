package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.ScopeStatement


interface GPContext{
    val device: GPDevice
    val disposed: Boolean
    val modules: GPModules

    fun compile(ast: ScopeStatement): Program
    fun compile(code: String) =
        compile(GPAst.parse(code, this))

    fun dispose()
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