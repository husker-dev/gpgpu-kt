package com.huskerdev.gpkt

import com.huskerdev.gpkt.engines.OpenGLAsyncDevice
import com.huskerdev.gpkt.engines.OpenGLSyncDevice

internal actual val defaultExpectedTypes = arrayOf(GPType.OpenGL)
internal actual val defaultExpectedDeviceId = 0

internal actual fun createSupportedSyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPSyncDevice? = requestedType.firstNotNullOfOrNull {
    when {
        it == GPType.OpenGL -> OpenGLSyncDevice()
        else -> null
    }
}

internal actual suspend fun createSupportedAsyncInstance(
    requestedDeviceId: Int,
    requestedType: Array<out GPType>
): GPAsyncDevice? = requestedType.firstNotNullOfOrNull {
    when {
        it == GPType.OpenGL -> OpenGLAsyncDevice()
        else -> null
    }
}