@file:Suppress("unused")
package com.huskerdev.gpkt.ast.utils

import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.objects.GPScope

fun scope(parentScope: GPScope, apply: (GPScope) -> Unit) =
    GPScope(parentScope.context, parentScope, parentScope.dictionary).apply(apply)

fun GPScope.toStatement(scope: GPScope) =
    ScopeStatement(scope, this)
