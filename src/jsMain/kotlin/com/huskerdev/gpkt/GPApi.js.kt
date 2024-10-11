package com.huskerdev.gpkt

import com.huskerdev.gpkt.apis.interpreter.InterpreterAsyncApi
import com.huskerdev.gpkt.apis.interpreter.InterpreterSyncApi
import com.huskerdev.gpkt.apis.js.JSAsyncApi
import com.huskerdev.gpkt.apis.js.JSSyncApi
import com.huskerdev.gpkt.apis.webgpu.WebGPU
import com.huskerdev.gpkt.apis.webgpu.WebGPUApi

internal actual val supportedApis = arrayOf(GPApiType.WebGPU, GPApiType.JS, GPApiType.Interpreter)

internal actual val defaultDeviceId = 0

internal actual val defaultSyncApisOrder: Array<GPApiType> =
    arrayOf(GPApiType.JS, GPApiType.Interpreter)

internal actual val defaultAsyncApisOrder: Array<GPApiType> =
    arrayOf(GPApiType.WebGPU, GPApiType.JS, GPApiType.Interpreter)

internal actual fun createSyncApiInstance(type: GPApiType): GPSyncApi? = when (type) {
    GPApiType.Interpreter -> InterpreterSyncApi()
    GPApiType.JS -> JSSyncApi()
    else -> null
}

internal actual suspend fun createAsyncApiInstance(type: GPApiType): GPAsyncApi? = when{
    type == GPApiType.WebGPU && WebGPU.supported -> WebGPUApi.create()
    type == GPApiType.Interpreter -> InterpreterAsyncApi()
    type == GPApiType.JS -> JSAsyncApi()
    else -> null
}