package com.huskerdev.gpkt.engines.cpu.objects

import com.huskerdev.gpkt.ast.types.Type

open class ExField(
    val type: Type,
    val value: ExValue?
)