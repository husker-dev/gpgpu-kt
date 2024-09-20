package com.huskerdev.gpkt.ast.lexer

import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.Type


val primitives = Type.entries.map { it.text }.toSet()
val modifiers = Modifiers.entries.map { it.text }.toSet()
val operatorTokens = Operator.entries.filter { it.token.isNotEmpty() } .map { it.token }.toSet()

val spacing = setOf(Char(10), Char(32), '\t')
val digits = setOf('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
val logical = setOf("true", "false")
val specials_separators = setOf("(", ")", "{", "}", "[", "]", ";", ",") + operatorTokens
val specials_keywords = setOf("if", "for", "while", "return", "break", "continue") + primitives + modifiers
const val longestSpecial = 7


data class Lexeme(
    val text: String,
    val type: Type,
    val lineIndex: Int,
    val inlineIndex: Int
) {
    enum class Type {
        NAME,
        SPECIAL,
        NUMBER,
        NUMBER_FLOATING_POINT,
        LOGICAL
    }
}