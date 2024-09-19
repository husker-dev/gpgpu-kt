package com.huskerdev.gpkt

import com.huskerdev.gpkt.cuda.Cuda
import com.huskerdev.gpkt.cuda.CudaEngine
import com.huskerdev.gpkt.opencl.OCLEngine


internal actual fun createSupportedInstance(vararg expectedEngine: GPType): GPEngine? {
    expectedEngine.forEach {
        when {
            it == GPType.OpenCL -> return OCLEngine()
            it == GPType.CUDA && Cuda.supported -> return CudaEngine()
        }
    }
    return null
}