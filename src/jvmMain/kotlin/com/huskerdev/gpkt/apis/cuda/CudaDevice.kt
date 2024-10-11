package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*
import jcuda.driver.CUdevice

abstract class CudaDevice(
    override val api: CudaApi,
    val peer: CUdevice,
    val cuda: Cuda = api.cuda
): GPDevice {
    override val name = cuda.getDeviceName(peer)
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