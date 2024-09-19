package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.Source
import jcuda.driver.CUdeviceptr

class CudaSource(
    private val cuda: Cuda,
    val ptr: CUdeviceptr,
    override val length: Int
): Source {
    override fun read() =
        cuda.read(ptr, length)

    override fun dealloc() {
        cuda.dealloc(ptr)
    }
}