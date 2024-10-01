package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import org.jocl.Pointer
import org.jocl.Sizeof
import org.jocl.cl_mem


abstract class OpenCLMemoryPointer<T>(
    private val typeSize: Int,
    private val wrapper: (T) -> Pointer
): MemoryPointer<T> {
    abstract val cl: OpenCL
    abstract val ptr: cl_mem

    override fun dealloc() =
        cl.dealloc(ptr)

    override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        cl.read(ptr, wrapper(dst),
            size = length.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize
        )
    }

    override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        cl.write(ptr, wrapper(src),
            size = length.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize
        )
    }
}

class CLFloatMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer<FloatArray>(Sizeof.cl_float, Pointer::to), FloatMemoryPointer

class CLDoubleMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer<DoubleArray>(Sizeof.cl_double, Pointer::to), DoubleMemoryPointer

class CLLongMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer<LongArray>(Sizeof.cl_long, Pointer::to), LongMemoryPointer

class CLIntMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer<IntArray>(Sizeof.cl_int, Pointer::to), IntMemoryPointer

class CLByteMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer<ByteArray>(Sizeof.cl_char, Pointer::to), ByteMemoryPointer