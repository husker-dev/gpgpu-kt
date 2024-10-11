package com.huskerdev.gpkt

import com.huskerdev.gpkt.apis.interpreter.InterpreterAsyncApi
import com.huskerdev.gpkt.apis.interpreter.InterpreterSyncApi
import com.huskerdev.gpkt.apis.opencl.OpenCL
import com.huskerdev.gpkt.apis.opencl.OpenCLAsyncApi
import com.huskerdev.gpkt.apis.opencl.OpenCLSyncApi

internal actual val defaultDeviceId = 0

internal actual val supportedApis: Array<GPApiType> =
    arrayOf(GPApiType.OpenCL, GPApiType.Interpreter)

internal actual val defaultSyncApisOrder: Array<GPApiType> =
    arrayOf(GPApiType.OpenCL, GPApiType.Interpreter)

internal actual val defaultAsyncApisOrder: Array<GPApiType> =
    arrayOf(GPApiType.OpenCL, GPApiType.Interpreter)

internal actual fun createSyncApiInstance(type: GPApiType): GPSyncApi? = when{
    type == GPApiType.OpenCL && OpenCL.supported -> OpenCLSyncApi()
    type == GPApiType.Interpreter -> InterpreterSyncApi()
    else -> null
}

internal actual suspend fun createAsyncApiInstance(type: GPApiType): GPAsyncApi? = when{
    type == GPApiType.OpenCL && OpenCL.supported -> OpenCLAsyncApi()
    type == GPApiType.Interpreter -> InterpreterAsyncApi()
    else -> null
}

