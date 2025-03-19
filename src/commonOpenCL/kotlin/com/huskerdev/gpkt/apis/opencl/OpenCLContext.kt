package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.GPScope


abstract class OpenCLContext(
    clDevice: OpenCLDevice,
): GPContext{
    override val device = clDevice

    val opencl = clDevice.opencl
    val peer = opencl.createContext(clDevice.platform, clDevice.peer)
    val commandQueue = clCreateCommandQueue(peer, clDevice.peer)

    override val allocated = arrayListOf<GPResource>()

    override val memory = opencl.getDeviceMemory(clDevice.peer)

    override var released = false
    override val modules = GPModules()

    override fun compile(ast: GPScope) =
        addResource(OpenCLProgram(this, ast))

    override fun release() {
        if(released) return
        allocated.toList().forEach(GPResource::release)
        clReleaseCommandQueue(commandQueue)
        opencl.disposeContext(peer)
        released = true
    }

    override fun releaseMemory(memory: MemoryPointer<*>) {
        allocated -= memory
        opencl.deallocMemory((memory as OpenCLMemoryPointer<*>).mem)
    }

    override fun releaseProgram(program: GPProgram) {
        allocated -= program
        opencl.deallocProgram((program as OpenCLProgram).program)
        opencl.deallocKernel(program.kernel)
    }

    protected fun <T: GPResource> addResource(memory: T): T{
        allocated += memory
        return memory
    }
}

class OpenCLSyncContext(
    clDevice: OpenCLDevice,
): OpenCLContext(clDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(CLSyncFloatMemoryPointer(this, array.size, usage, opencl.wrapFloats(peer, array, usage)))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(CLSyncFloatMemoryPointer(this, length, usage, opencl.allocate(peer, Float.SIZE_BYTES * length, usage)))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(CLSyncIntMemoryPointer(this, array.size, usage, opencl.wrapInts(peer, array, usage)))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(CLSyncIntMemoryPointer(this, length, usage, opencl.allocate(peer, Int.SIZE_BYTES * length, usage)))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(CLSyncByteMemoryPointer(this, array.size, usage, opencl.wrapBytes(peer, array, usage)))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(CLSyncByteMemoryPointer(this, length, usage, opencl.allocate(peer, length, usage)))
}

class OpenCLAsyncContext(
    clDevice: OpenCLDevice,
): OpenCLContext(clDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(CLAsyncFloatMemoryPointer(this, array.size, usage, opencl.wrapFloats(peer, array, usage)))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(CLAsyncFloatMemoryPointer(this, length, usage, opencl.allocate(peer, Float.SIZE_BYTES * length, usage)))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(CLAsyncIntMemoryPointer(this, array.size, usage, opencl.wrapInts(peer, array, usage)))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(CLAsyncIntMemoryPointer(this, length, usage, opencl.allocate(peer, Int.SIZE_BYTES * length, usage)))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(CLAsyncByteMemoryPointer(this, array.size, usage, opencl.wrapBytes(peer, array, usage)))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(CLAsyncByteMemoryPointer(this, length, usage, opencl.allocate(peer, length, usage)))
}

