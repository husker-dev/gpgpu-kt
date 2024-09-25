package com.huskerdev.gpkt.utils

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.objects.Field

// Example output: __kernel void main(float*a, int b){
fun appendCFunctionHeader(
    buffer: StringBuilder,
    modifiers: List<String>,
    type: String,
    name: String,
    args: List<String>
){
    appendCModifiers(buffer, modifiers)
    buffer.append(type).append(" ").append(name).append("(")
    args.joinTo(buffer, separator = ",")
    buffer.append("){")
}

// Example output: __constant int a, b = 1;
fun appendCFieldHeader(
    buffer: StringBuilder,
    modifiers: List<String>,
    type: String,
    fields: List<Field>,
    expressionGen: (Expression) -> Unit
){
    appendCModifiers(buffer, modifiers)
    buffer.append(type).append(" ")
    fields.forEachIndexed { i, field ->
        buffer.append(field.name)
        if(field.initialExpression != null){
            buffer.append("=")
            expressionGen(field.initialExpression)
        }
        if(i == fields.lastIndex) buffer.append(";")
        else buffer.append(",")
    }
}

// Example output: __constant in
fun appendCModifiers(
    buffer: StringBuilder,
    modifiers: List<String>
) {
    modifiers.joinTo(buffer, separator = " ")
    if(modifiers.isNotEmpty())
        buffer.append(" ")
}
