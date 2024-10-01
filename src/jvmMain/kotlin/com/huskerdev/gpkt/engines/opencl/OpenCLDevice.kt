package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import org.jocl.CL.*
import org.jocl.Pointer
import org.jocl.Sizeof

class OpenCLDevice(
    requestedDeviceId: Int
): GPDevice(GPType.OpenCL) {
    private val cl = OpenCL(requestedDeviceId)

    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CLFloatMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_float, usage.toCL()))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CLFloatMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_float * length, usage.toCL()))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CLDoubleMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_double, usage.toCL()))

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CLDoubleMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_double * length, usage.toCL()))

    override fun allocLong(array: LongArray, usage: MemoryUsage) =
        CLLongMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_long, usage.toCL()))

    override fun allocLong(length: Int, usage: MemoryUsage) =
        CLLongMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_long * length, usage.toCL()))

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CLIntMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_int, usage.toCL()))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CLIntMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_int * length, usage.toCL()))

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CLByteMemoryPointer(cl, array.size, usage,
            cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_char, usage.toCL()))

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CLByteMemoryPointer(cl, length, usage,
            cl.allocate(Sizeof.cl_char * length, usage.toCL()))

    override fun compile(ast: ScopeStatement) =
        OpenCLProgram(cl, ast)

    private fun MemoryUsage.toCL() = when(this){
        MemoryUsage.READ_ONLY -> CL_MEM_READ_ONLY
        MemoryUsage.WRITE_ONLY -> CL_MEM_WRITE_ONLY
        MemoryUsage.READ_WRITE -> CL_MEM_READ_WRITE
    }
}