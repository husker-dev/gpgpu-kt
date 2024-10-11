package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*

abstract class MetalApi: GPApi {
    val metal = createMetal()
    override val type = GPApiType.Metal
}

class MetalSyncApi: MetalApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> = metal.copyAllDevices().map { peer ->
        MetalSyncDevice(this, peer)
    }.toTypedArray()
}

class MetalAsyncApi: MetalApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> = metal.copyAllDevices().map { peer ->
        MetalAsyncDevice(this, peer)
    }.toTypedArray()
}