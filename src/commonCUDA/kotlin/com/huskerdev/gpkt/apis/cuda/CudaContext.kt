package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.GPScope


abstract class CudaContext(
    cudaDevice: CudaDevice
): GPContext{
    override val device = cudaDevice
    val peer = Cuda.createContext(cudaDevice.peer)

    override val allocated = arrayListOf<GPResource>()

    override val memory = Cuda.getDeviceMemory(cudaDevice.peer)

    override var released = false
    override val modules = GPModules()

    override fun compile(ast: GPScope) =
        addResource(CudaProgram(this, ast))

    override fun release() {
        if(released) return
        allocated.toList().forEach(GPResource::release)
        Cuda.dispose(peer)
        released = true
    }

    override fun releaseMemory(memory: MemoryPointer<*>) {
        allocated -= memory
        Cuda.dealloc(peer, (memory as CudaMemoryPointer<*>).ptr)
    }

    override fun releaseProgram(program: GPProgram) {
        allocated -= program
        Cuda.unloadModule((program as CudaProgram).module)
    }

    protected fun <T: GPResource> addResource(memory: T): T{
        allocated += memory
        return memory
    }
}

class CudaSyncContext(
    cudaDevice: CudaDevice
): CudaContext(cudaDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(CudaSyncFloatMemoryPointer(this, array.size, usage, Cuda.wrapFloats(peer, array)))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(CudaSyncFloatMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Float.SIZE_BYTES)))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(CudaSyncIntMemoryPointer(this, array.size, usage, Cuda.wrapInts(peer, array)))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(CudaSyncIntMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Int.SIZE_BYTES)))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(CudaSyncByteMemoryPointer(this, array.size, usage, Cuda.wrapBytes(peer, array)))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(CudaSyncByteMemoryPointer(this, length, usage, Cuda.alloc(peer, length)))
}

class CudaAsyncContext(
    cudaDevice: CudaDevice
): CudaContext(cudaDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(CudaAsyncFloatMemoryPointer(this, array.size, usage, Cuda.wrapFloats(peer, array)))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(CudaAsyncFloatMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Float.SIZE_BYTES)))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(CudaAsyncIntMemoryPointer(this, array.size, usage, Cuda.wrapInts(peer, array)))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(CudaAsyncIntMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Int.SIZE_BYTES)))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(CudaAsyncByteMemoryPointer(this, array.size, usage, Cuda.wrapBytes(peer, array)))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(CudaAsyncByteMemoryPointer(this, length, usage, Cuda.alloc(peer, length)))
}