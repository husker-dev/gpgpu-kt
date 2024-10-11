package com.huskerdev.gpkt.apis.jdk

import com.huskerdev.gpkt.*

abstract class JavacDevice: GPDevice {
    override val name = "CPU"
    override val isAccelerated = false
}

class JavacSyncDevice(
    override val api: GPApi
): JavacDevice(), GPSyncDevice {
    override fun createContext() =
        JavacSyncContext(this)
}

class JavacAsyncDevice(
    override val api: GPApi
): JavacDevice(), GPAsyncDevice {
    override suspend fun createContext() =
        JavacAsyncContext(this)
}