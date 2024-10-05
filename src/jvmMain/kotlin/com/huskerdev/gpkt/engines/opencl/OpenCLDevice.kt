package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import org.jocl.Sizeof


abstract class CLDeviceBase(requestedDeviceId: Int): GPDeviceBase{
    protected val cl = OpenCL(requestedDeviceId)

    override val type = GPType.OpenCL
    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement) =
        OpenCLProgram(cl, ast)
}

class OpenCLSyncDevice(
    requestedDeviceId: Int
): CLDeviceBase(requestedDeviceId), GPSyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(cl, array.size, usage, cl.wrapFloats(array, usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(cl, length, usage, cl.allocate(Sizeof.cl_float * length, usage))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        CLSyncDoubleMemoryPointer(cl, array.size, usage, cl.wrapDoubles(array, usage))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        CLSyncDoubleMemoryPointer(cl, length, usage, cl.allocate(Sizeof.cl_double * length, usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(cl, array.size, usage, cl.wrapInts(array, usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(cl, length, usage, cl.allocate(Sizeof.cl_int * length, usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(cl, array.size, usage, cl.wrapBytes(array, usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(cl, length, usage, cl.allocate(length, usage))
}

class OpenCLAsyncDevice(
    requestedDeviceId: Int
): CLDeviceBase(requestedDeviceId), GPAsyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(cl, array.size, usage, cl.wrapFloats(array, usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(cl, length, usage, cl.allocate(Sizeof.cl_float * length, usage))

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        CLAsyncDoubleMemoryPointer(cl, array.size, usage, cl.wrapDoubles(array, usage))

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        CLAsyncDoubleMemoryPointer(cl, length, usage, cl.allocate(Sizeof.cl_double * length, usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(cl, array.size, usage, cl.wrapInts(array, usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(cl, length, usage, cl.allocate(Sizeof.cl_int * length, usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(cl, array.size, usage, cl.wrapBytes(array, usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(cl, length, usage, cl.allocate(length, usage))
}

