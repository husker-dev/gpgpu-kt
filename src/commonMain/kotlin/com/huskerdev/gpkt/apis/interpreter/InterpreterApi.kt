package com.huskerdev.gpkt.apis.interpreter

import com.huskerdev.gpkt.*

abstract class InterpreterApi: GPApi {
    override val type = GPApiType.Interpreter
}

class InterpreterSyncApi: InterpreterApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> =
        arrayOf(InterpreterSyncDevice(this))
}

class InterpreterAsyncApi: InterpreterApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> =
        arrayOf(InterpreterAsyncDevice(this))
}