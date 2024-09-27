package com.huskerdev.gpkt.ast.types


enum class Modifiers(val text: String){
    IN("in"),
    OUT("out"),
    CONST("const")
    ;
    companion object {
        val map = entries.associateBy { it.text }
    }
}