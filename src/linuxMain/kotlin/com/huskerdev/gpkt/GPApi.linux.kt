package com.huskerdev.gpkt

import com.huskerdev.gpkt.apis.cuda.Cuda
import com.huskerdev.gpkt.apis.cuda.CudaAsyncApi
import com.huskerdev.gpkt.apis.cuda.CudaSyncApi
import com.huskerdev.gpkt.apis.interpreter.InterpreterAsyncApi
import com.huskerdev.gpkt.apis.interpreter.InterpreterSyncApi

internal actual val supportedApis: Array<GPApiType>
    = arrayOf(GPApiType.CUDA, GPApiType.Interpreter)

internal actual val defaultSyncApisOrder: Array<GPApiType>
    = arrayOf(GPApiType.CUDA, GPApiType.Interpreter)

internal actual val defaultAsyncApisOrder: Array<GPApiType>
    = arrayOf(GPApiType.CUDA, GPApiType.Interpreter)

internal actual val defaultDeviceId: Int = 0

internal actual fun createSyncApiInstance(type: GPApiType): GPSyncApi? = when {
    type == GPApiType.CUDA && Cuda.supported -> CudaSyncApi()
    type == GPApiType.Interpreter -> InterpreterSyncApi()
    else -> throw UnsupportedOperationException()
}

internal actual suspend fun createAsyncApiInstance(type: GPApiType): GPAsyncApi? = when {
    type == GPApiType.CUDA && Cuda.supported -> CudaAsyncApi()
    type == GPApiType.Interpreter -> InterpreterAsyncApi()
    else -> throw UnsupportedOperationException()
}