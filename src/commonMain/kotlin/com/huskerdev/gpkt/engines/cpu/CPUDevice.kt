package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.ScopeStatement

open class CPUSyncDevice: CPUSyncDeviceBase(GPType.Interpreter) {
    override fun compile(ast: ScopeStatement) =
        CPUProgram(ast)
}

open class CPUAsyncDevice: CPUAsyncDeviceBase(GPType.Interpreter) {
    override fun compile(ast: ScopeStatement) =
        CPUProgram(ast)
}