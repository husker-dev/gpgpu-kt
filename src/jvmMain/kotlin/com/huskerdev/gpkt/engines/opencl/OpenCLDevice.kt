package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.Scope
import org.jocl.Pointer
import org.jocl.Sizeof

class OpenCLDevice(
    requestedDeviceId: Int
): GPDevice(GPType.OpenCL) {
    private val cl = OpenCL(requestedDeviceId)

    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true

    override fun allocFloat(array: FloatArray) =
        CLFloatMemoryPointer(cl, cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_float), array.size)

    override fun allocFloat(length: Int) =
        CLFloatMemoryPointer(cl, cl.allocate(Sizeof.cl_float * length), length)

    override fun allocDouble(array: DoubleArray) =
        CLDoubleMemoryPointer(cl, cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_double), array.size)

    override fun allocDouble(length: Int) =
        CLDoubleMemoryPointer(cl, cl.allocate(Sizeof.cl_double * length), length)

    override fun allocLong(array: LongArray) =
        CLLongMemoryPointer(cl, cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_long), array.size)

    override fun allocLong(length: Int) =
        CLLongMemoryPointer(cl, cl.allocate(Sizeof.cl_long * length), length)

    override fun allocInt(array: IntArray) =
        CLIntMemoryPointer(cl, cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_int), array.size)

    override fun allocInt(length: Int) =
        CLIntMemoryPointer(cl, cl.allocate(Sizeof.cl_int * length), length)

    override fun allocByte(array: ByteArray) =
        CLByteMemoryPointer(cl, cl.allocate(Pointer.to(array), array.size.toLong() * Sizeof.cl_char), array.size)

    override fun allocByte(length: Int) =
        CLByteMemoryPointer(cl, cl.allocate(Sizeof.cl_char * length), length)

    override fun compile(ast: Scope) =
        OpenCLProgram(cl, ast)

}