package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import jcuda.Sizeof

abstract class CudaContext(
    cudaDevice: CudaDevice
): GPContext{
    override val device = cudaDevice
    val cuda = cudaDevice.cuda
    val peer = cuda.createContext(cudaDevice.peer)

    override var disposed = false
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement) =
        CudaProgram(this, ast)

    override fun dispose() {
        if(disposed) return
        disposed = true
        cuda.dispose(peer)
    }
}

class CudaSyncContext(
    cudaDevice: CudaDevice
): CudaContext(cudaDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(this, array.size, usage, cuda.wrapFloats(peer, array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(this, length, usage, cuda.alloc(peer, length * Sizeof.FLOAT))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(this, array.size, usage, cuda.wrapInts(peer, array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(this, length, usage, cuda.alloc(peer, length * Sizeof.INT))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(this, array.size, usage, cuda.wrapBytes(peer, array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(this, length, usage, cuda.alloc(peer, length))
}

class CudaAsyncContext(
    cudaDevice: CudaDevice
): CudaContext(cudaDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(this, array.size, usage, cuda.wrapFloats(peer, array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(this, length, usage, cuda.alloc(peer, length * Sizeof.FLOAT))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(this, array.size, usage, cuda.wrapInts(peer, array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(this, length, usage, cuda.alloc(peer, length * Sizeof.INT))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(this, array.size, usage, cuda.wrapBytes(peer, array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(this, length, usage, cuda.alloc(peer, length))
}