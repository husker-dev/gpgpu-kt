package com.huskerdev.gpkt.apis.webgpu

import com.huskerdev.gpkt.*

class WebGPUDevice(
    override val api: WebGPUApi
): GPAsyncDevice {
    val webgpu = api.webgpu

    override val name = webgpu.getAdapterName(api.adapterPeer)
    override val isAccelerated = true

    override suspend fun createContext(): WebGPUAsyncContext {
        val device = webgpu.requestDevice(api.adapterPeer)
        return WebGPUAsyncContext(
            this,
            webgpu.requestDevice(api.adapterPeer),
            webgpu.createCommandEncoder(device)
        )
    }
}