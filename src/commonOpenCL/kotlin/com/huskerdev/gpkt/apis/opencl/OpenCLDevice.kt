package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.GPAsyncDevice
import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPSyncDevice

abstract class OpenCLDevice(
    override val api: OpenCLApi,
    val peer: CLDeviceId,
    val opencl: OpenCL = api.opencl
): GPDevice {
    val platform = opencl.clGetPlatformIDs()[0]

    override val name = opencl.getDeviceName(peer)
    override val isAccelerated = true
}

class OpenCLSyncDevice(
    api: OpenCLApi,
    peer: CLDeviceId,
): OpenCLDevice(api, peer), GPSyncDevice {
    override fun createContext() =
        OpenCLSyncContext(this)
}

class OpenCLAsyncDevice(
    api: OpenCLApi,
    peer: CLDeviceId,
): OpenCLDevice(api, peer), GPAsyncDevice {
    override suspend fun createContext() =
        OpenCLAsyncContext(this)
}