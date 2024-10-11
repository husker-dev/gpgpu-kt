package com.huskerdev.gpkt.apis.interpreter

import com.huskerdev.gpkt.GPAsyncDevice
import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPSyncDevice

abstract class InterpreterDevice: GPDevice {
    override val name = "CPU"
    override val isAccelerated = false
}

class InterpreterSyncDevice(
    override val api: InterpreterApi
): InterpreterDevice(), GPSyncDevice {
    override fun createContext() =
        InterpreterSyncContext(this)
}

class InterpreterAsyncDevice(
    override val api: InterpreterApi
): InterpreterDevice(), GPAsyncDevice {
    override suspend fun createContext() =
        InterpreterAsyncContext(this)
}