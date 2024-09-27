package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import org.jocl.Pointer
import org.jocl.Sizeof
import org.jocl.cl_mem


interface OpenCLMemoryPointer{
    val ptr: cl_mem
}

class CLFloatMemoryPointer(
    private val cl: OpenCL,
    override val ptr: cl_mem,
    override val length: Int
): OpenCLMemoryPointer, FloatMemoryPointer {
    override fun read() = FloatArray(length).apply {
        cl.read(ptr, length.toLong() * Sizeof.cl_float, Pointer.to(this))
    }
    override fun dealloc() = cl.dealloc(ptr)
}

class CLDoubleMemoryPointer(
    private val cl: OpenCL,
    override val ptr: cl_mem,
    override val length: Int
): OpenCLMemoryPointer, DoubleMemoryPointer {
    override fun read() = DoubleArray(length).apply {
        cl.read(ptr, length.toLong() * Sizeof.cl_double, Pointer.to(this))
    }
    override fun dealloc() = cl.dealloc(ptr)
}

class CLLongMemoryPointer(
    private val cl: OpenCL,
    override val ptr: cl_mem,
    override val length: Int
): OpenCLMemoryPointer, LongMemoryPointer {
    override fun read() = LongArray(length).apply {
        cl.read(ptr, length.toLong() * Sizeof.cl_long, Pointer.to(this))
    }
    override fun dealloc() = cl.dealloc(ptr)
}

class CLIntMemoryPointer(
    private val cl: OpenCL,
    override val ptr: cl_mem,
    override val length: Int
): OpenCLMemoryPointer, IntMemoryPointer {
    override fun read() = IntArray(length).apply {
        cl.read(ptr, length.toLong() * Sizeof.cl_int, Pointer.to(this))
    }
    override fun dealloc() = cl.dealloc(ptr)
}

class CLByteMemoryPointer(
    private val cl: OpenCL,
    override val ptr: cl_mem,
    override val length: Int
): OpenCLMemoryPointer, ByteMemoryPointer {
    override fun read() = ByteArray(length).apply {
        cl.read(ptr, length.toLong() * Sizeof.cl_char, Pointer.to(this))
    }
    override fun dealloc() = cl.dealloc(ptr)
}