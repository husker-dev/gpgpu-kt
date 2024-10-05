package com.huskerdev.gpkt.engines.js

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.engines.cpu.*

class JSSyncDevice: CPUSyncDevice() {
    override val type = GPType.JS

    override fun compile(ast: ScopeStatement) =
        JSProgram(ast)
}

class JSAsyncDevice: CPUAsyncDevice(){
    override val type = GPType.JS

    override fun compile(ast: ScopeStatement) =
        JSProgram(ast)
}