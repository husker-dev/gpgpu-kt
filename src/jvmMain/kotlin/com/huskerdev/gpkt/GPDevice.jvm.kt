package com.huskerdev.gpkt

import com.huskerdev.gpkt.engines.cuda.Cuda
import com.huskerdev.gpkt.engines.cuda.CudaDevice
import com.huskerdev.gpkt.engines.jdk.ClassCompiler
import com.huskerdev.gpkt.engines.jdk.JavacDevice
import com.huskerdev.gpkt.engines.opencl.OpenCLDevice
import com.huskerdev.gpkt.engines.opencl.OpenCL

internal actual val defaultExpectedTypes: Array<GPType> =
    System.getenv().getOrDefault("gp.order", "cuda,opencl,javac,interpreter")
        .split(",").map {
            GPType.mapped.getOrElse(it) { throw Exception("Unknown GPType: '$it'") }
        }.toTypedArray()

internal actual val defaultExpectedDeviceId: Int =
    System.getenv().getOrDefault("gp.index", "0").toInt()

internal actual fun createSupportedInstance(requestedDeviceId: Int, vararg requestedType: GPType): GPDevice? {
    requestedType.forEach {
        when {
            it == GPType.Javac && ClassCompiler.supported -> return JavacDevice()
            it == GPType.OpenCL && OpenCL.supported -> return OpenCLDevice(requestedDeviceId)
            it == GPType.CUDA && Cuda.supported -> return CudaDevice(requestedDeviceId)
        }
    }
    return null
}

