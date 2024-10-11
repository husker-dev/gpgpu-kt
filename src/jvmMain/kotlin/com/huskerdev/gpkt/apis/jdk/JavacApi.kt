package com.huskerdev.gpkt.apis.jdk

import com.huskerdev.gpkt.*

abstract class JavacApi: GPApi {
    override val type = GPApiType.Javac
}

class JavacSyncApi: JavacApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> =
        arrayOf(JavacSyncDevice(this))
}

class JavacAsyncApi: JavacApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> =
        arrayOf(JavacAsyncDevice(this))
}

