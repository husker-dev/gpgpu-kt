package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.objects.Scope


class GPLibraries(
    private val contextDevice: GPDevice
) {
    val ast = hashMapOf<String, Scope>()

    fun add(name: String, code: String){
        ast[name] = GPAst.parse(code, contextDevice)
    }
}

