package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.engines.cpu.*


class JavacSyncDevice: CPUSyncDevice() {
    override val type = GPType.Javac

    override fun compile(ast: ScopeStatement) =
        JavacProgram(ast)
}

class JavacAsyncDevice: CPUAsyncDevice() {
    override val type = GPType.Javac

    override fun compile(ast: ScopeStatement) =
        JavacProgram(ast)
}