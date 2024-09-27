package com.huskerdev.gpkt.ast.types


enum class Modifiers(val text: String){
    EXTERNAL("external"),
    CONST("const")
    ;
    companion object {
        val map = entries.associateBy { it.text }
    }
}