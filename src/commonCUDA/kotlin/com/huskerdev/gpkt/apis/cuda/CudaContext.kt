package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.GPScope


abstract class CudaContext(
    cudaDevice: CudaDevice
): GPContext{
    override val device = cudaDevice
    val peer = Cuda.createContext(cudaDevice.peer)

    override var disposed = false
    override val modules = GPModules()

    override fun compile(ast: GPScope) =
        CudaProgram(this, ast)

    override fun dispose() {
        if(disposed) return
        disposed = true
        Cuda.dispose(peer)
    }
}

class CudaSyncContext(
    cudaDevice: CudaDevice
): CudaContext(cudaDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(this, array.size, usage, Cuda.wrapFloats(peer, array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(this, array.size, usage, Cuda.wrapInts(peer, array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(this, array.size, usage, Cuda.wrapBytes(peer, array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(this, length, usage, Cuda.alloc(peer, length))
}

class CudaAsyncContext(
    cudaDevice: CudaDevice
): CudaContext(cudaDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(this, array.size, usage, Cuda.wrapFloats(peer, array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(this, array.size, usage, Cuda.wrapInts(peer, array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(this, length, usage, Cuda.alloc(peer, length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(this, array.size, usage, Cuda.wrapBytes(peer, array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(this, length, usage, Cuda.alloc(peer, length))
}