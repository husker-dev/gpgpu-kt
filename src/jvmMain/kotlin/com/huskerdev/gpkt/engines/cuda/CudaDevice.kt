package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import jcuda.Sizeof

abstract class CudaDeviceBase(
    requestedDeviceId: Int
): GPDeviceBase{
    protected val cuda = Cuda(requestedDeviceId)

    override val type = GPType.CUDA
    override val id = cuda.deviceId
    override val name = cuda.deviceName
    override val isGPU = true
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement) =
        CudaProgram(cuda, ast)
}

class CudaSyncDevice(
    requestedDeviceId: Int
): CudaDeviceBase(requestedDeviceId), GPSyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(cuda, array.size, usage, cuda.wrapFloats(array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CudaSyncFloatMemoryPointer(cuda, length, usage, cuda.alloc(length * Sizeof.FLOAT))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        CudaSyncDoubleMemoryPointer(cuda, array.size, usage, cuda.wrapDoubles(array))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        CudaSyncDoubleMemoryPointer(cuda, length, usage, cuda.alloc(length * Sizeof.DOUBLE))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(cuda, array.size, usage, cuda.wrapInts(array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CudaSyncIntMemoryPointer(cuda, length, usage, cuda.alloc(length * Sizeof.INT))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(cuda, array.size, usage, cuda.wrapBytes(array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CudaSyncByteMemoryPointer(cuda, length, usage, cuda.alloc(length))
}

class CudaAsyncDevice(
    requestedDeviceId: Int
): CudaDeviceBase(requestedDeviceId), GPAsyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(cuda, array.size, usage, cuda.wrapFloats(array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CudaAsyncFloatMemoryPointer(cuda, length, usage, cuda.alloc(length * Sizeof.FLOAT))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        CudaAsyncDoubleMemoryPointer(cuda, array.size, usage, cuda.wrapDoubles(array))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        CudaAsyncDoubleMemoryPointer(cuda, length, usage, cuda.alloc(length * Sizeof.DOUBLE))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(cuda, array.size, usage, cuda.wrapInts(array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CudaAsyncIntMemoryPointer(cuda, length, usage, cuda.alloc(length * Sizeof.INT))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(cuda, array.size, usage, cuda.wrapBytes(array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CudaAsyncByteMemoryPointer(cuda, length, usage, cuda.alloc(length))
}