package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import org.jocl.CL.*
import org.jocl.Pointer
import org.jocl.Sizeof

private fun MemoryUsage.toCL() = when(this){
    MemoryUsage.READ_ONLY -> CL_MEM_READ_ONLY
    MemoryUsage.WRITE_ONLY -> CL_MEM_WRITE_ONLY
    MemoryUsage.READ_WRITE -> CL_MEM_READ_WRITE
}

class OpenCLSyncDevice(
    requestedDeviceId: Int
): GPSyncDevice(GPType.OpenCL) {
    private val cl = OpenCL(requestedDeviceId)

    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_float, usage.toCL()))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CLSyncFloatMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_float * length, usage.toCL()))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CLSyncDoubleMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_double, usage.toCL()))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CLSyncDoubleMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_double * length, usage.toCL()))

    override fun allocLong(array: LongArray, usage: MemoryUsage) =
        CLSyncLongMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_long, usage.toCL()))

    override fun allocLong(length: Int, usage: MemoryUsage) =
        CLSyncLongMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_long * length, usage.toCL()))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_int, usage.toCL()))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CLSyncIntMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_int * length, usage.toCL()))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_char, usage.toCL()))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CLSyncByteMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_char * length, usage.toCL()))

    override fun compile(ast: ScopeStatement) =
        OpenCLProgram(cl, ast)
}

class OpenCLAsyncDevice(
    requestedDeviceId: Int
): GPAsyncDevice(GPType.OpenCL) {
    private val cl = OpenCL(requestedDeviceId)

    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_float, usage.toCL()))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CLAsyncFloatMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_float * length, usage.toCL()))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CLAsyncDoubleMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_double, usage.toCL()))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CLAsyncDoubleMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_double * length, usage.toCL()))

    override fun allocLong(array: LongArray, usage: MemoryUsage) =
        CLAsyncLongMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_long, usage.toCL()))

    override fun allocLong(length: Int, usage: MemoryUsage) =
        CLAsyncLongMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_long * length, usage.toCL()))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_int, usage.toCL()))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CLAsyncIntMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_int * length, usage.toCL()))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_char, usage.toCL()))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CLAsyncByteMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_char * length, usage.toCL()))

    override fun compile(ast: ScopeStatement) =
        OpenCLProgram(cl, ast)


}

