package com.huskerdev.gpkt.engines.js

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.engines.cpu.*

class JSSyncDevice: CPUSyncDeviceBase(GPType.JS) {
    override fun compile(ast: ScopeStatement) =
        JSProgram(ast)
}

class JSAsyncDevice: CPUAsyncDeviceBase(GPType.JS){
    override fun compile(ast: ScopeStatement) =
        JSProgram(ast)
}