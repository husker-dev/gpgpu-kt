package com.huskerdev.gpkt



class GPModules {
    val ast = hashMapOf<String, () -> String>()

    fun add(name: String, block: () -> String){
        ast[name] = block
    }

    operator fun set(name: String, block: () -> String) =
        add(name, block)

    operator fun get(name: String) =
        ast[name]

    operator fun contains(name: String) =
        ast.contains(name)
}

