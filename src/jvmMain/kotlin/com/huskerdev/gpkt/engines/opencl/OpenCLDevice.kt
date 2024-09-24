package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class OpenCLDevice(
    requestedDeviceId: Int
): GPDevice(GPType.OpenCL) {
    private val cl = OpenCL(requestedDeviceId)

    override val id = cl.deviceId
    override val name = cl.deviceName
    override val isGPU = true

    override fun alloc(array: FloatArray) =
        OpenCLSource(cl, cl.allocate(array), array.size)

    override fun alloc(length: Int) =
        OpenCLSource(cl, cl.allocate(length), length)

    override fun compile(ast: Scope) =
        OpenCLProgram(cl, ast)

}