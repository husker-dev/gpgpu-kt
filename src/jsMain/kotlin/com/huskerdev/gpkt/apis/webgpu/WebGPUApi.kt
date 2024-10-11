package com.huskerdev.gpkt.apis.webgpu

import com.huskerdev.gpkt.GPApiType
import com.huskerdev.gpkt.GPAsyncApi
import com.huskerdev.gpkt.GPAsyncDevice

class WebGPUApi(
    val webgpu: WebGPU,
    val adapterPeer: dynamic
): GPAsyncApi {
    companion object {
        suspend fun create(): WebGPUApi{
            val webgpu = WebGPU()
            return WebGPUApi(webgpu, webgpu.requestAdapter())
        }
    }

    override val type = GPApiType.WebGPU

    override val devices: Array<GPAsyncDevice> =
        arrayOf(WebGPUDevice(this))
}