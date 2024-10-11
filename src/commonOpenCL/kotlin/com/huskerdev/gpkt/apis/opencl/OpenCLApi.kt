package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.*

abstract class OpenCLApi: GPApi {
    val opencl = createCL()

    override val type = GPApiType.OpenCL
}

class OpenCLSyncApi: OpenCLApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> = opencl.getDevices().map { peer ->
        OpenCLSyncDevice(this, peer)
    }.toTypedArray()
}

class OpenCLAsyncApi: OpenCLApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> = opencl.getDevices().map { peer ->
        OpenCLAsyncDevice(this, peer)
    }.toTypedArray()
}