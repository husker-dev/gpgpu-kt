package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.GPEngine
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class CPUEngine: GPEngine(GPType.Interpreter) {

    override fun compile(ast: Scope) =
        CPUProgram(ast)

    override fun alloc(array: FloatArray) =
        CPUSource(array)

    override fun alloc(length: Int) =
        CPUSource(FloatArray(length))
}