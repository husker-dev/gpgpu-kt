package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class JavacDevice: GPDevice(GPType.Javac) {
    override val id = 0
    override val name = "CPU"
    override val isGPU = false

    override fun compile(ast: Scope) =
        JavacProgram(ast)

    override fun alloc(array: FloatArray) =
        JavacSource(array.clone())

    override fun alloc(length: Int) =
        JavacSource(FloatArray(length))
}