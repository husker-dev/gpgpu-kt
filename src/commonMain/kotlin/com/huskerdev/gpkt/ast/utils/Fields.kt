@file:Suppress("unused")
package com.huskerdev.gpkt.ast.utils

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.FieldExpression
import com.huskerdev.gpkt.ast.FieldStatement
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.PrimitiveType

fun threadLocalField(name: String, type: PrimitiveType, value: Expression? = null) =
    GPField(name, arrayListOf(Modifiers.THREADLOCAL), type, value)

fun externalField(name: String, type: PrimitiveType, value: Expression? = null) =
    GPField(name, arrayListOf(Modifiers.EXTERNAL), type, value)

fun GPField.toStatement(scope: GPScope) =
    FieldStatement(scope, listOf(this))

fun GPField.toExpr() =
    FieldExpression(null, this)