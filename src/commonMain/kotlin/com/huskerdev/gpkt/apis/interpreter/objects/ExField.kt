package com.huskerdev.gpkt.apis.interpreter.objects

import com.huskerdev.gpkt.ast.types.PrimitiveType

open class ExField(
    val type: PrimitiveType,
    val value: ExValue?
)