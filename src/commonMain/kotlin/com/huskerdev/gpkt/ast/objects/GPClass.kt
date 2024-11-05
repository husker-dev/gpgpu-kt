package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.types.*


class GPClass(
    val scope: GPScope,
    val name: String,
    val type: ClassType,
    val variables: Map<String, GPField>,
    val variablesTypes: List<PrimitiveType>,
    val body: ScopeStatement?,
    val obfName: String,
    val implements: List<String>
)
