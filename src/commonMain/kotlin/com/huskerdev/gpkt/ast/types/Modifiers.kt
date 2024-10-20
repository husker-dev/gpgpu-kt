package com.huskerdev.gpkt.ast.types


enum class Modifiers(val text: String){
    EXTERNAL("extern"),
    CONST("const"),
    READONLY("readonly"),   // For extern with read-only access
    THREADLOCAL("threadlocal")          // For variables in global scope to make them thread-local
    ;
    companion object {
        val map = entries.associateBy { it.text }
    }
}