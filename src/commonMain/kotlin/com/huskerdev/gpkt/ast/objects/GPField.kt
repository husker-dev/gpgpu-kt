package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.types.FLOAT
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.PrimitiveType


class GPField(
    var name: String,
    val obfName: String = name,
    val modifiers: MutableList<Modifiers>,
    val type: PrimitiveType,
    var initialExpression: Expression? = null
) {
    constructor(
        name: String,
        modifiers: MutableList<Modifiers>,
        type: PrimitiveType,
        initialExpression: Expression? = null
    ): this(name, name, modifiers, type, initialExpression)
    constructor(name: String, type: PrimitiveType): this(name, name, mutableListOf(), type, null)
    constructor(name: String, type: PrimitiveType, initialExpression: Expression): this(name, name, mutableListOf(), type, initialExpression)

    val isExtern
        get() = Modifiers.EXTERNAL in modifiers
    val isLocal
        get() = Modifiers.THREADLOCAL in modifiers
    val isConstant
        get() = Modifiers.CONST in modifiers
    val isReadonly
        get() = Modifiers.READONLY in modifiers

    fun clone(scope: GPScope) =
        GPField(name, obfName, modifiers.toMutableList(), type, initialExpression?.clone(scope))
}

val predefinedMathFields = hashMapOf(
    fieldPair("PI", FLOAT),
    fieldPair("E", FLOAT),
    fieldPair("NaN", FLOAT),
    fieldPair("FLOAT_MAX", FLOAT),
    fieldPair("FLOAT_MIN", FLOAT),
    fieldPair("INT_MAX", FLOAT),
    fieldPair("INT_MIN", FLOAT),
)

val allPredefinedFields = predefinedMathFields

private fun fieldPair(name: String, type: PrimitiveType) =
    name to GPField(name, type)