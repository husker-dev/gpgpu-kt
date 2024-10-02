package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.engines.cpu.*

class JavacSyncDevice: CPUSyncDeviceBase(GPType.Javac) {
    override fun compile(ast: ScopeStatement) =
        JavacProgram(ast)
}

class JavacAsyncDevice: CPUAsyncDeviceBase(GPType.Javac) {
    override fun compile(ast: ScopeStatement) =
        JavacProgram(ast)
}