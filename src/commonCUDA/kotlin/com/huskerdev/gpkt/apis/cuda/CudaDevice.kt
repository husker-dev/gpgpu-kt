package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*

abstract class CudaDevice(
    override val api: CudaApi,
    val peer: CUdevice
): GPDevice {
    override val name = Cuda.getDeviceName(peer)
    override val isAccelerated = true
}

class CudaSyncDevice(
    api: CudaApi,
    peer: CUdevice
): CudaDevice(api, peer), GPSyncDevice {
    override fun createContext() =
        CudaSyncContext(this)
}

class CudaAsyncDevice(
    api: CudaApi,
    peer: CUdevice
): CudaDevice(api, peer), GPAsyncDevice {
    override suspend fun createContext() =
        CudaAsyncContext(this)
}