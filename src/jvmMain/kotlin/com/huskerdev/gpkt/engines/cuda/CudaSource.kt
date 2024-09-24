package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.Source
import jcuda.Pointer
import jcuda.driver.CUdeviceptr

class CudaSource(
    private val cuda: Cuda,
    private val devicePtr: CUdeviceptr,
    override val length: Int
): Source {
    val ptr: Pointer = Pointer.to(devicePtr)

    override fun read() =
        cuda.read(devicePtr, length)

    override fun dealloc() {
        cuda.dealloc(devicePtr)
    }
}