package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.types.FLOAT
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.PrimitiveType


class GPField(
    val name: String,
    val modifiers: MutableList<Modifiers>,
    val type: PrimitiveType,
    var initialExpression: Expression? = null
) {
    constructor(name: String, type: PrimitiveType): this(name, mutableListOf(), type, null)
    constructor(name: String, type: PrimitiveType, initialExpression: Expression): this(name, mutableListOf(), type, initialExpression)

    val isExtern
        get() = Modifiers.EXTERNAL in modifiers
    val isLocal
        get() = Modifiers.THREADLOCAL in modifiers
    val isConstant
        get() = Modifiers.CONST in modifiers
    val isReadonly
        get() = Modifiers.READONLY in modifiers
}

val predefinedMathFields = hashMapOf(
    fieldPair("PI", FLOAT),
    fieldPair("E", FLOAT),
)

val allPredefinedFields = predefinedMathFields

private fun fieldPair(name: String, type: PrimitiveType) =
    name to GPField(name, type)