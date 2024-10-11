package com.huskerdev.gpkt.apis.js

import com.huskerdev.gpkt.GPAsyncDevice
import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPSyncDevice

abstract class JSDevice(
    override val api: JSApi
): GPDevice {
    override val name = "CPU"
    override val isAccelerated = false
}

class JSSyncDevice(
    api: JSApi
): JSDevice(api), GPSyncDevice {
    override fun createContext() =
        JSSyncContext(this)
}

class JSAsyncDevice(
    api: JSApi
): JSDevice(api), GPAsyncDevice {
    override suspend fun createContext() =
        JSAsyncContext(this)
}