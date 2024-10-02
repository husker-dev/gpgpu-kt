package com.huskerdev.gpkt.ast.types


enum class Modifiers(val text: String){
    EXTERNAL("extern"),
    CONST("const"),
    READONLY("readonly")
    ;
    companion object {
        val map = entries.associateBy { it.text }
    }
}