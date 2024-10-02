package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.engines.cpu.CPUAsyncDevice
import com.huskerdev.gpkt.engines.cpu.CPUSyncDevice


internal expect val defaultExpectedTypes: Array<GPType>
internal expect val defaultExpectedDeviceId: Int

internal expect fun createSupportedSyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPSyncDevice?

internal expect suspend fun createSupportedAsyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPAsyncDevice?


abstract class GPDeviceBase(
    val type: GPType
){
    abstract val id: Int
    abstract val name: String
    abstract val isGPU: Boolean

    val modules = GPModules(this)
}


abstract class GPSyncDevice(
    type: GPType
): GPDeviceBase(type) {
    companion object {
        fun create(
            requestedDeviceId: Int = defaultExpectedDeviceId,
            vararg requestedType: GPType = defaultExpectedTypes,
        ) = createSupportedSyncInstance(requestedDeviceId, requestedType) ?:
        if(GPType.Interpreter in requestedType) CPUSyncDevice() else null
    }

    fun compile(code: String) =
        compile(GPAst.parse(code, this))

    abstract fun allocFloat(array: FloatArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncFloatMemoryPointer
    abstract fun allocFloat(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncFloatMemoryPointer

    abstract fun allocDouble(array: DoubleArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncDoubleMemoryPointer
    abstract fun allocDouble(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncDoubleMemoryPointer

    abstract fun allocLong(array: LongArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncLongMemoryPointer
    abstract fun allocLong(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncLongMemoryPointer

    abstract fun allocInt(array: IntArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncIntMemoryPointer
    abstract fun allocInt(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncIntMemoryPointer

    abstract fun allocByte(array: ByteArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncByteMemoryPointer
    abstract fun allocByte(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncByteMemoryPointer

    abstract fun compile(ast: ScopeStatement): Program
}


abstract class GPAsyncDevice(type: GPType): GPDeviceBase(type) {
    companion object {
        suspend fun create(
            requestedDeviceId: Int = defaultExpectedDeviceId,
            vararg requestedType: GPType = defaultExpectedTypes,
        ) = createSupportedAsyncInstance(requestedDeviceId, requestedType) ?:
            if(GPType.Interpreter in requestedType) CPUAsyncDevice() else null
    }

    fun compile(code: String) =
        compile(GPAst.parse(code, this))

    abstract fun allocFloat(array: FloatArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncFloatMemoryPointer
    abstract fun allocFloat(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncFloatMemoryPointer

    abstract fun allocDouble(array: DoubleArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncDoubleMemoryPointer
    abstract fun allocDouble(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncDoubleMemoryPointer

    abstract fun allocLong(array: LongArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncLongMemoryPointer
    abstract fun allocLong(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncLongMemoryPointer

    abstract fun allocInt(array: IntArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncIntMemoryPointer
    abstract fun allocInt(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncIntMemoryPointer

    abstract fun allocByte(array: ByteArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncByteMemoryPointer
    abstract fun allocByte(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncByteMemoryPointer

    abstract fun compile(ast: ScopeStatement): Program
}