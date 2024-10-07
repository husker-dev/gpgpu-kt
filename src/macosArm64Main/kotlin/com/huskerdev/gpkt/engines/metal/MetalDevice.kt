package com.huskerdev.gpkt.engines.metal

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement

abstract class MetalDeviceBase(
    requestedDeviceId: Int
): GPDeviceBase{
    protected val metal = Metal(requestedDeviceId)

    override val type = GPType.Metal
    override val id = metal.deviceId
    override val name = metal.name
    override val isGPU = true
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement) =
        MetalProgram(metal, ast)
}

class MetalSyncDevice(
    requestedDeviceId: Int
): MetalDeviceBase(requestedDeviceId), GPSyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        MetalSyncFloatMemoryPointer(metal, array.size, usage, metal.wrapFloats(array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        MetalSyncFloatMemoryPointer(metal, length, usage, metal.createBuffer(length * Float.SIZE_BYTES))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        MetalSyncDoubleMemoryPointer(metal, array.size, usage, metal.wrapDoubles(array))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        MetalSyncDoubleMemoryPointer(metal, length, usage, metal.createBuffer(length * Double.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        MetalSyncIntMemoryPointer(metal, array.size, usage, metal.wrapInts(array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        MetalSyncIntMemoryPointer(metal, length, usage, metal.createBuffer(length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        MetalSyncByteMemoryPointer(metal, array.size, usage, metal.wrapBytes(array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        MetalSyncByteMemoryPointer(metal, length, usage, metal.createBuffer(length))
}

class MetalAsyncDevice(
    requestedDeviceId: Int
): MetalDeviceBase(requestedDeviceId), GPAsyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        MetalAsyncFloatMemoryPointer(metal, array.size, usage, metal.wrapFloats(array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        MetalAsyncFloatMemoryPointer(metal, length, usage, metal.createBuffer(length * Float.SIZE_BYTES))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        MetalAsyncDoubleMemoryPointer(metal, array.size, usage, metal.wrapDoubles(array))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        MetalAsyncDoubleMemoryPointer(metal, length, usage, metal.createBuffer(length * Double.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        MetalAsyncIntMemoryPointer(metal, array.size, usage, metal.wrapInts(array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        MetalAsyncIntMemoryPointer(metal, length, usage, metal.createBuffer(length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        MetalAsyncByteMemoryPointer(metal, array.size, usage, metal.wrapBytes(array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        MetalAsyncByteMemoryPointer(metal, length, usage, metal.createBuffer(length))

}