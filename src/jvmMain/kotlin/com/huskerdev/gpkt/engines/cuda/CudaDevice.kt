package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class CudaDevice(
    requestedDeviceId: Int
): GPDevice(GPType.CUDA) {
    private val cuda = Cuda(requestedDeviceId)

    override val id = cuda.deviceId
    override val name = cuda.deviceName
    override val isGPU = true

    override fun alloc(array: FloatArray) =
        CudaSource(cuda, cuda.alloc(array), array.size)

    override fun alloc(length: Int) =
        CudaSource(cuda, cuda.alloc(length), length)

    override fun compile(ast: Scope) =
        CudaProgram(cuda, ast)
}