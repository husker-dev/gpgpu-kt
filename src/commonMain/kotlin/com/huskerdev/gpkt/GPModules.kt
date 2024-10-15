package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.ScopeStatement


class GPModules(
    private val contextDevice: GPContext
) {
    val ast = hashMapOf<String, ScopeStatement>()

    fun add(name: String, code: String){
        ast[name] = GPAst.parse(code, contextDevice, false)
    }

    operator fun set(name: String, code: String) =
        add(name, code)

    operator fun get(name: String) =
        ast[name]

}

