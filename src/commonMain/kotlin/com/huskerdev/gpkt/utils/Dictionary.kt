package com.huskerdev.gpkt.utils


private val badWords = setOf(
    // All CPP keywords without underscores
    "alignas", "alignof", "and", "asm", "auto", "bitand", "bitor",
    "bool", "break", "case", "catch", "char", "class", "compl",
    "concept", "const", "consteval", "constexpr", "constinit",
    "continue", "co_await", "decltype", "default", "delete", "do",
    "double", "else", "enum", "explicit", "export", "extern", "false",
    "float", "for", "friend", "goto", "if", "inline", "int", "long",
    "mutable", "namespace", "new", "noexcept", "not", "nullptr",
    "operator", "or", "private", "protected", "public", "reflexpr",
    "register", "requires", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "synchronized", "template", "this",
    "throw", "true", "try", "typedef", "typeid", "typename", "union",
    "unsigned", "using", "virtual", "void", "volatile", "while", "xor",

    // Java keywords (not included earlier)
    "abstract", "assert", "boolean", "byte", "exports", "extends",
    "final", "finally", "implements", "import", "instanceof",
    "interface", "module", "native", "package", "strictfp", "super",
    "throws", "transient", "var", "volatile",

    // JS keywords (not included earlier)
    "debugger", "let"
)

class Dictionary(
    private var index: Int = 0
){
    companion object {
        private val alphabet = "abcdefghijklmnopqrstuvwxyz".toCharArray()
        var includeName = false
    }

    fun nextWord(prefix: String): String {
        while(true){
            val name = if(includeName)
                "${prefix}_${generate()}" else generate()
            if(name in badWords) continue
            return name
        }
    }

    private fun generate(): String{
        val chars = arrayListOf<Char>()

        var curIndex = index++
        if(curIndex >= alphabet.size) {
            while(curIndex >= alphabet.size) {
                val i = curIndex % alphabet.size
                chars += alphabet[i]
                curIndex /= alphabet.size
            }
            chars += alphabet[curIndex % alphabet.size - 1]
        }else
            chars += alphabet[curIndex % alphabet.size]

        return chars.toCharArray().concatToString()
    }
}