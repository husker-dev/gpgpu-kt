package com.huskerdev.gpkt

import com.huskerdev.gpkt.engines.cuda.Cuda
import com.huskerdev.gpkt.engines.cuda.CudaEngine
import com.huskerdev.gpkt.engines.jdk.JavacEngine
import com.huskerdev.gpkt.engines.opencl.OCLEngine


internal actual fun createSupportedInstance(vararg expectedEngine: GPType): GPEngine? {
    expectedEngine.forEach {
        when {
            it == GPType.Javac -> return JavacEngine()
            it == GPType.OpenCL -> return OCLEngine()
            it == GPType.CUDA && Cuda.supported -> return CudaEngine()
        }
    }
    return null
}