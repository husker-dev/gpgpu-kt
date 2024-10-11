package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.GPAsyncDevice
import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPSyncDevice

abstract class MetalDevice(
    override val api: MetalApi,
    val peer: MTLDevice,
    val metal: Metal = api.metal
): GPDevice {
    override val name = metal.getDeviceName(peer)
    override val isAccelerated = true
}

class MetalSyncDevice(
    api: MetalApi,
    peer: MTLDevice
): MetalDevice(api, peer), GPSyncDevice {
    override fun createContext() =
        MetalSyncContext(this)
}

class MetalAsyncDevice(
    api: MetalApi,
    peer: MTLDevice
): MetalDevice(api, peer), GPAsyncDevice {
    override suspend fun createContext() =
        MetalAsyncContext(this)
}