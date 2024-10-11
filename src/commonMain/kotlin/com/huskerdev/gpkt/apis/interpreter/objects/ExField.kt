package com.huskerdev.gpkt.apis.interpreter.objects

import com.huskerdev.gpkt.ast.types.Type

open class ExField(
    val type: Type,
    val value: ExValue?
)