package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type


class Function(
    val scope: Scope?,
    val name: String,
    val modifiers: List<Modifiers>,
    val returnType: Type
){
    val arguments = mutableListOf<Field>()
    val argumentsTypes = mutableListOf<Type>()
    lateinit var body: ScopeStatement

    constructor(
        returnType: Type,
        name: String,
        vararg argumentTypes: Pair<String, Type>
    ): this(null, name, emptyList(), returnType){
        argumentTypes.forEach { pair ->
            addArgument(Field(pair.first, mutableListOf(), pair.second))
        }
    }

    fun canAcceptArguments(types: List<Type>): Boolean{
        if(types.size != argumentsTypes.size)
            return false
        argumentsTypes.forEachIndexed { index, argType ->
            val type = types[index]
            if(argType != type && !Type.canAssignNumbers(argType, type))
                return false
        }
        return true
    }

    fun addArgument(argument: Field){
        arguments += argument
        argumentsTypes += argument.type
    }
}

val predefinedMathFunctions = hashMapOf(
    funPair("abs", Type.FLOAT, "a" to Type.FLOAT),
    funPair("acos", Type.FLOAT, "angle" to Type.FLOAT),
    funPair("asin", Type.FLOAT, "angle" to Type.FLOAT),
    funPair("atan", Type.FLOAT, "angle" to Type.FLOAT),
    funPair("atan2", Type.FLOAT, "y" to Type.FLOAT, "x" to Type.FLOAT),
    funPair("cbrt", Type.FLOAT, "a" to Type.FLOAT),
    funPair("ceil", Type.FLOAT, "a" to Type.FLOAT),
    funPair("cos", Type.FLOAT, "angle" to Type.FLOAT),
    funPair("cosh", Type.FLOAT, "x" to Type.FLOAT),
    funPair("exp", Type.FLOAT, "a" to Type.FLOAT),
    funPair("expm1", Type.FLOAT, "x" to Type.FLOAT),
    funPair("floor", Type.FLOAT, "a" to Type.FLOAT),
    funPair("hypot", Type.FLOAT, "x" to Type.FLOAT, "y" to Type.FLOAT),
    funPair("log", Type.FLOAT, "a" to Type.FLOAT),
    funPair("log10", Type.FLOAT, "a" to Type.FLOAT),
    funPair("max", Type.FLOAT, "a" to Type.FLOAT, "b" to Type.FLOAT),
    funPair("min", Type.FLOAT, "a" to Type.FLOAT, "b" to Type.FLOAT),
    funPair("pow", Type.FLOAT, "a" to Type.FLOAT, "b" to Type.INT),
    funPair("round", Type.FLOAT, "a" to Type.FLOAT),
    funPair("sin", Type.FLOAT, "a" to Type.FLOAT),
    funPair("sinh", Type.FLOAT, "x" to Type.FLOAT),
    funPair("sqrt", Type.FLOAT, "a" to Type.FLOAT),
    funPair("tan", Type.FLOAT, "a" to Type.FLOAT),
    funPair("tanh", Type.FLOAT, "x" to Type.FLOAT),
)

val allPredefinedFunctions = predefinedMathFunctions

private fun funPair(name: String, type: Type, vararg arguments: Pair<String, Type>) =
    name to Function(type, name, *arguments)