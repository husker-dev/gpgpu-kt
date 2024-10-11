package com.huskerdev.gpkt

import com.huskerdev.gpkt.apis.interpreter.InterpreterAsyncApi
import com.huskerdev.gpkt.apis.interpreter.InterpreterSyncApi
import com.huskerdev.gpkt.apis.metal.MetalAsyncApi
import com.huskerdev.gpkt.apis.metal.MetalSyncApi

internal actual val supportedApis = arrayOf(GPApiType.Metal, GPApiType.Interpreter)

internal actual val defaultDeviceId = 0

internal actual val defaultSyncApisOrder: Array<GPApiType> =
    arrayOf(GPApiType.Metal, GPApiType.Interpreter)

internal actual val defaultAsyncApisOrder: Array<GPApiType> =
    arrayOf(GPApiType.Metal, GPApiType.Interpreter)

internal actual fun createSyncApiInstance(type: GPApiType): GPSyncApi? = when(type){
    GPApiType.Interpreter -> InterpreterSyncApi()
    GPApiType.Metal -> MetalSyncApi()
    else -> null
}

internal actual suspend fun createAsyncApiInstance(type: GPApiType): GPAsyncApi? = when(type){
    GPApiType.Interpreter -> InterpreterAsyncApi()
    GPApiType.Metal -> MetalAsyncApi()
    else -> null
}