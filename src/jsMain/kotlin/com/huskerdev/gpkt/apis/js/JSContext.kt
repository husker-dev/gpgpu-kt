package com.huskerdev.gpkt.apis.js

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.apis.interpreter.InterpreterAsyncContext
import com.huskerdev.gpkt.apis.interpreter.InterpreterSyncContext
import com.huskerdev.gpkt.ast.objects.GPScope

class JSSyncContext(
    device: GPDevice
): InterpreterSyncContext(device) {
    override fun compile(ast: GPScope) =
        JSProgram(ast)
}

class JSAsyncContext(
    device: GPDevice
): InterpreterAsyncContext(device){
    override fun compile(ast: GPScope) =
        JSProgram(ast)
}