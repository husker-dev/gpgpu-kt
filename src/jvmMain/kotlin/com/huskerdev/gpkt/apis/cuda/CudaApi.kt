package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.*

abstract class CudaApi: GPApi {
    val cuda = Cuda()

    override val type = GPApiType.CUDA
}

class CudaSyncApi: CudaApi(), GPSyncApi {
    override val devices: Array<GPSyncDevice> = cuda.getDevices().map {
        CudaSyncDevice(this, it)
    }.toTypedArray()
}

class CudaAsyncApi: CudaApi(), GPAsyncApi {
    override val devices: Array<GPAsyncDevice> = cuda.getDevices().map {
        CudaAsyncDevice(this, it)
    }.toTypedArray()
}