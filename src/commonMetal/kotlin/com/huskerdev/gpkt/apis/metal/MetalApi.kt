package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*

abstract class MetalApi: GPApi {
    override val type = GPApiType.Metal
}

class MetalSyncApi: MetalApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> = mtlCopyAllDevices().map { peer ->
        MetalSyncDevice(this, peer)
    }.toTypedArray()
}

class MetalAsyncApi: MetalApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> = mtlCopyAllDevices().map { peer ->
        MetalAsyncDevice(this, peer)
    }.toTypedArray()
}