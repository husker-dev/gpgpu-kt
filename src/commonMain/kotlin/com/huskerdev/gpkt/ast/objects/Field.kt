package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.ast.lexer.Lexeme


class Field(
    val lexeme: Lexeme,
    val name: String,
    val modifiers: MutableList<Modifiers>,
    val type: Type,
    val initialExpression: Expression? = null
) {
    val isConstant = Modifiers.CONST in modifiers
}