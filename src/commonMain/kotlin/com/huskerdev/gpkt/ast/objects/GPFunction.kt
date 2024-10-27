package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.types.FLOAT
import com.huskerdev.gpkt.ast.types.INT
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.PrimitiveType


class GPFunction(
    val scope: GPScope?,
    val name: String,
    val modifiers: List<Modifiers>,
    val returnType: PrimitiveType
){
    val arguments = mutableListOf<GPField>()
    val argumentsTypes = mutableListOf<PrimitiveType>()
    var body: ScopeStatement? = null

    constructor(
        returnType: PrimitiveType,
        name: String,
        vararg argumentTypes: Pair<String, PrimitiveType>
    ): this(null, name, emptyList(), returnType){
        argumentTypes.forEach { pair ->
            addArgument(GPField(pair.first, mutableListOf(), pair.second))
        }
    }

    fun canAcceptArguments(types: List<PrimitiveType>): Boolean{
        if(types.size != argumentsTypes.size)
            return false
        argumentsTypes.forEachIndexed { index, argType ->
            val type = types[index]
            if(argType != type && !PrimitiveType.canAssignNumbers(argType, type))
                return false
        }
        return true
    }

    fun addArgument(argument: GPField){
        arguments += argument
        argumentsTypes += argument.type
    }
}

val predefinedMathFunctions = hashMapOf(
    funPair("abs", FLOAT, "a" to FLOAT),
    funPair("acos", FLOAT, "angle" to FLOAT),
    funPair("asin", FLOAT, "angle" to FLOAT),
    funPair("atan", FLOAT, "angle" to FLOAT),
    funPair("atan2", FLOAT, "y" to FLOAT, "x" to FLOAT),
    funPair("cbrt", FLOAT, "a" to FLOAT),
    funPair("ceil", FLOAT, "a" to FLOAT),
    funPair("cos", FLOAT, "angle" to FLOAT),
    funPair("cosh", FLOAT, "x" to FLOAT),
    funPair("exp", FLOAT, "a" to FLOAT),
    funPair("expm1", FLOAT, "x" to FLOAT),
    funPair("floor", FLOAT, "a" to FLOAT),
    funPair("hypot", FLOAT, "x" to FLOAT, "y" to FLOAT),
    funPair("log", FLOAT, "a" to FLOAT),
    funPair("log10", FLOAT, "a" to FLOAT),
    funPair("max", FLOAT, "a" to FLOAT, "b" to FLOAT),
    funPair("min", FLOAT, "a" to FLOAT, "b" to FLOAT),
    funPair("pow", FLOAT, "a" to FLOAT, "b" to INT),
    funPair("round", FLOAT, "a" to FLOAT),
    funPair("sin", FLOAT, "a" to FLOAT),
    funPair("sinh", FLOAT, "x" to FLOAT),
    funPair("sqrt", FLOAT, "a" to FLOAT),
    funPair("tan", FLOAT, "a" to FLOAT),
    funPair("tanh", FLOAT, "x" to FLOAT),
)

val allPredefinedFunctions = predefinedMathFunctions

private fun funPair(name: String, type: PrimitiveType, vararg arguments: Pair<String, PrimitiveType>) =
    name to GPFunction(type, name, *arguments)