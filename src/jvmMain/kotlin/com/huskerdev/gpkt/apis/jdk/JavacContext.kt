package com.huskerdev.gpkt.apis.jdk

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.apis.interpreter.InterpreterAsyncContext
import com.huskerdev.gpkt.apis.interpreter.InterpreterSyncContext
import com.huskerdev.gpkt.ast.objects.GPScope

class JavacSyncContext(
    device: GPDevice
): InterpreterSyncContext(device) {
    override fun compile(ast: GPScope) =
        JavacProgram(this, ast)
}

class JavacAsyncContext(
    device: GPDevice
): InterpreterAsyncContext(device) {
    override fun compile(ast: GPScope) =
        JavacProgram(this, ast)
}