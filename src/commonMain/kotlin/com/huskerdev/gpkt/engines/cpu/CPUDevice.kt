package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class CPUDevice: GPDevice(GPType.Interpreter) {
    override val id = 0
    override val name = "CPU"
    override val isGPU = false

    override fun compile(ast: Scope) =
        CPUProgram(ast)

    override fun alloc(array: FloatArray) =
        CPUSource(array)

    override fun alloc(length: Int) =
        CPUSource(FloatArray(length))
}