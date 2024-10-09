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


interface GPDeviceBase{
    val type: GPType
    val id: Int
    val name: String
    val isGPU: Boolean

    val modules: GPModules

    fun compile(ast: ScopeStatement): Program
    fun compile(code: String) =
        compile(GPAst.parse(code, this))
}

interface GPSyncDevice: GPDeviceBase {
    companion object {
        fun create(
            requestedDeviceId: Int = defaultExpectedDeviceId,
            vararg requestedType: GPType = defaultExpectedTypes,
        ) = createSupportedSyncInstance(requestedDeviceId, requestedType) ?:
            if(GPType.Interpreter in requestedType) CPUSyncDevice() else null
    }

    fun wrapFloats(array: FloatArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncFloatMemoryPointer
    fun allocFloats(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncFloatMemoryPointer

    fun wrapInts(array: IntArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncIntMemoryPointer
    fun allocInts(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncIntMemoryPointer

    fun wrapBytes(array: ByteArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncByteMemoryPointer
    fun allocBytes(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): SyncByteMemoryPointer
}


interface GPAsyncDevice: GPDeviceBase {
    companion object {
        suspend fun create(
            requestedDeviceId: Int = defaultExpectedDeviceId,
            vararg requestedType: GPType = defaultExpectedTypes,
        ) = createSupportedAsyncInstance(requestedDeviceId, requestedType) ?:
            if(GPType.Interpreter in requestedType) CPUAsyncDevice() else null
    }

    fun wrapFloats(array: FloatArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncFloatMemoryPointer
    fun allocFloats(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncFloatMemoryPointer

    fun wrapInts(array: IntArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncIntMemoryPointer
    fun allocInts(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncIntMemoryPointer

    fun wrapBytes(array: ByteArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncByteMemoryPointer
    fun allocBytes(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): AsyncByteMemoryPointer
}