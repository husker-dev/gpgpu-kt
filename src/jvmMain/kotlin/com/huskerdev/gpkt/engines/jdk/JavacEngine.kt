package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.GPEngine
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class JavacEngine: GPEngine(GPType.Javac) {
    override fun compile(ast: Scope) =
        JavacProgram(ast)

    override fun alloc(array: FloatArray) =
        JavacSource(array.clone())

    override fun alloc(length: Int) =
        JavacSource(FloatArray(length))
}