package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.Source
import org.jocl.cl_mem


class OCLSource(
    private val cl: OpenCL,
    val data: cl_mem,
    override val length: Int
): Source {
    override fun read() =
        cl.read(data, length)

    override fun dealloc() {
        cl.dealloc(data)
    }
}