package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.ScopeStatement


class GPModules(
    private val contextDevice: GPDevice
) {
    val ast = hashMapOf<String, ScopeStatement>()

    fun add(name: String, code: String){
        ast[name] = GPAst.parse(code, contextDevice)
    }
}

