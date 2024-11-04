package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*

abstract class CudaApi: GPApi {
    override val type = GPApiType.CUDA
}

class CudaSyncApi: CudaApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> = Cuda.getDevices().map {
        CudaSyncDevice(this, it)
    }.toTypedArray()
}

class CudaAsyncApi: CudaApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> = Cuda.getDevices().map {
        CudaAsyncDevice(this, it)
    }.toTypedArray()
}