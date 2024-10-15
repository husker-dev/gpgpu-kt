package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type


class Field(
    val name: String,
    val modifiers: MutableList<Modifiers>,
    val type: Type,
    var initialExpression: Expression? = null
) {
    constructor(name: String, type: Type): this(name, mutableListOf(), type, null)
    constructor(name: String, type: Type, initialExpression: Expression): this(name, mutableListOf(), type, initialExpression)

    val isConstant = Modifiers.CONST in modifiers
    val isReadonly = Modifiers.READONLY in modifiers
}

val predefinedMathFields = hashMapOf(
    fieldPair("PI", Type.FLOAT),
    fieldPair("E", Type.FLOAT),
)

val allPredefinedFields = predefinedMathFields

private fun fieldPair(name: String, type: Type) =
    name to Field(name, type)