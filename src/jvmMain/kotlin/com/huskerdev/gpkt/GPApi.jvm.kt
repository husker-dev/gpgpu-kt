package com.huskerdev.gpkt

import com.huskerdev.gpkt.apis.cuda.Cuda
import com.huskerdev.gpkt.apis.cuda.CudaAsyncApi
import com.huskerdev.gpkt.apis.cuda.CudaSyncApi
import com.huskerdev.gpkt.apis.interpreter.InterpreterAsyncApi
import com.huskerdev.gpkt.apis.interpreter.InterpreterSyncApi
import com.huskerdev.gpkt.apis.jdk.ClassCompiler
import com.huskerdev.gpkt.apis.jdk.JavacAsyncApi
import com.huskerdev.gpkt.apis.jdk.JavacSyncApi
import com.huskerdev.gpkt.apis.metal.MetalSyncApi
import com.huskerdev.gpkt.apis.metal.metalSupported
import com.huskerdev.gpkt.apis.opencl.OpenCL
import com.huskerdev.gpkt.apis.opencl.OpenCLAsyncApi
import com.huskerdev.gpkt.apis.opencl.OpenCLSyncApi

internal actual val supportedApis =
    arrayOf(GPApiType.CUDA, GPApiType.OpenCL, GPApiType.Metal, GPApiType.Javac, GPApiType.Interpreter)

internal actual val defaultDeviceId: Int =
    System.getenv().getOrDefault("gp.index", "0").toInt()

internal actual val defaultSyncApisOrder: Array<GPApiType> =
    System.getenv().getOrDefault("gp.sync.order", "cuda,opencl,metal,javac,interpreter")
        .split(",").map {
            GPApiType.mapped.getOrElse(it) { throw Exception("Unknown GPType: '$it'") }
        }.toTypedArray()

internal actual val defaultAsyncApisOrder: Array<GPApiType> =
    System.getenv().getOrDefault("gp.async.order", "cuda,opencl,metal,javac,interpreter")
        .split(",").map {
            GPApiType.mapped.getOrElse(it) { throw Exception("Unknown GPType: '$it'") }
        }.toTypedArray()

internal actual fun createSyncApiInstance(type: GPApiType): GPSyncApi? = when{
    type == GPApiType.CUDA && Cuda.supported -> CudaSyncApi()
    type == GPApiType.OpenCL && OpenCL.supported -> OpenCLSyncApi()
    type == GPApiType.Metal && metalSupported -> MetalSyncApi()
    type == GPApiType.Javac && ClassCompiler.supported -> JavacSyncApi()
    type == GPApiType.Interpreter -> InterpreterSyncApi()
    else -> null
}

internal actual suspend fun createAsyncApiInstance(type: GPApiType): GPAsyncApi? = when{
    type == GPApiType.CUDA && Cuda.supported -> CudaAsyncApi()
    type == GPApiType.OpenCL && OpenCL.supported -> OpenCLAsyncApi()
    type == GPApiType.Javac && ClassCompiler.supported -> JavacAsyncApi()
    type == GPApiType.Interpreter -> InterpreterAsyncApi()
    else -> null
}