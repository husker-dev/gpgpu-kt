package com.huskerdev.gpkt.ast.lexer

import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.primitivesMap


val primitives = primitivesMap.map { it.key }.toSet()
val modifiers = Modifiers.entries.map { it.text }.toSet()
val operatorTokens = Operator.entries.filter { it.token.isNotEmpty() } .map { it.token }.toSet()

val spacing = setOf(Char(10), Char(32), '\t')
val digits = setOf('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
val digitsHex = setOf('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f')
val logical = setOf("true", "false")
val specials_separators = setOf("(", ")", "{", "}", "[", "]", ";", ",") + operatorTokens
val specials_keywords = setOf("if", "for", "while", "return", "break", "continue", "import") + primitives + modifiers
const val longestSpecial = 7


data class Lexeme(
    val text: String,
    val type: Type,
    val lineIndex: Int = 0,
    val inlineIndex: Int = 0
) {
    enum class Type {
        NAME,
        SPECIAL,
        INT,
        LONG,
        BYTE,
        FLOAT,
        LOGICAL
    }
}