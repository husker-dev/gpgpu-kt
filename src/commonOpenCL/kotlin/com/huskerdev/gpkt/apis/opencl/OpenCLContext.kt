package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement


abstract class OpenCLContext(
    clDevice: OpenCLDevice,
): GPContext{
    override val device = clDevice

    val opencl = clDevice.opencl
    val peer = opencl.createContext(clDevice.platform, clDevice.peer)
    val commandQueue = clCreateCommandQueue(peer, clDevice.peer)

    override var disposed = true
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement) =
        OpenCLProgram(this, ast)

    override fun dispose() {
        if(disposed) return
        disposed = true
        opencl.disposeContext(peer)
    }
}

class OpenCLSyncContext(
    clDevice: OpenCLDevice,
): OpenCLContext(clDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(this, array.size, usage, opencl.wrapFloats(peer, array, usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(this, length, usage, opencl.allocate(peer, Float.SIZE_BYTES * length, usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(this, array.size, usage, opencl.wrapInts(peer, array, usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(this, length, usage, opencl.allocate(peer, Int.SIZE_BYTES * length, usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(this, array.size, usage, opencl.wrapBytes(peer, array, usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(this, length, usage, opencl.allocate(peer, length, usage))
}

class OpenCLAsyncContext(
    clDevice: OpenCLDevice,
): OpenCLContext(clDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(this, array.size, usage, opencl.wrapFloats(peer, array, usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(this, length, usage, opencl.allocate(peer, Float.SIZE_BYTES * length, usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(this, array.size, usage, opencl.wrapInts(peer, array, usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(this, length, usage, opencl.allocate(peer, Int.SIZE_BYTES * length, usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(this, array.size, usage, opencl.wrapBytes(peer, array, usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(this, length, usage, opencl.allocate(peer, length, usage))
}

