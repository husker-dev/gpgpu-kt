package com.huskerdev.gpkt.apis.js

import com.huskerdev.gpkt.*

abstract class JSApi: GPApi {
    override val type = GPApiType.JS
}

class JSSyncApi: JSApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> = arrayOf(JSSyncDevice(this))
}

class JSAsyncApi: JSApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> = arrayOf(JSAsyncDevice(this))
}