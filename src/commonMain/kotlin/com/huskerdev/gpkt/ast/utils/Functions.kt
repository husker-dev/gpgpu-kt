@file:Suppress("unused")
package com.huskerdev.gpkt.ast.utils

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.FunctionCallExpression
import com.huskerdev.gpkt.ast.objects.GPFunction

fun GPFunction.callExpr(vararg args: Expression) =
    FunctionCallExpression(
        obj = null,
        function = this,
        arguments = args.toList()
    )