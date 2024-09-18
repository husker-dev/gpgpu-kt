package com.huskerdev.gpkt.opencl

import com.huskerdev.gpkt.Source
import org.jocl.cl_mem


class OCLSource: Source {
    private val cl: OpenCL
    val data: cl_mem
    override val length: Int

    constructor(cl: OpenCL, array: FloatArray){
        this.cl = cl
        data = cl.allocate(array)
        length = array.size
    }

    constructor(cl: OpenCL, length: Int){
        this.cl = cl
        data = cl.allocate(length)
        this.length = length
    }

    override fun read() =
        cl.read(data, length)

    fun dealloc() {
        cl.dealloc(data)
    }
}