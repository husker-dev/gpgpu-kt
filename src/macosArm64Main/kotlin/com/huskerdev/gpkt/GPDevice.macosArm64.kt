package com.huskerdev.gpkt

import com.huskerdev.gpkt.engines.metal.MetalAsyncDevice
import com.huskerdev.gpkt.engines.metal.MetalSyncDevice

internal actual val defaultExpectedTypes = arrayOf(GPType.Metal, GPType.OpenCL, GPType.Interpreter)
internal actual val defaultExpectedDeviceId = 0

internal actual fun createSupportedSyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPSyncDevice? = requestedType.firstNotNullOfOrNull {
    when(it){
        GPType.Metal -> MetalSyncDevice(requestedDeviceId)
        else -> null
    }
}

internal actual suspend fun createSupportedAsyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPAsyncDevice? = requestedType.firstNotNullOfOrNull {
    when(it){
        GPType.Metal -> MetalAsyncDevice(requestedDeviceId)
        else -> null
    }
}