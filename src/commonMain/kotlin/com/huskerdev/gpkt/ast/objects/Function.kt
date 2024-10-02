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
    funPair("abs", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("acos", Type.DOUBLE, "angle" to Type.DOUBLE),
    funPair("asin", Type.DOUBLE, "angle" to Type.DOUBLE),
    funPair("atan", Type.DOUBLE, "angle" to Type.DOUBLE),
    funPair("atan2", Type.DOUBLE, "y" to Type.DOUBLE, "x" to Type.DOUBLE),
    funPair("cbrt", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("ceil", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("cos", Type.DOUBLE, "angle" to Type.DOUBLE),
    funPair("cosh", Type.DOUBLE, "x" to Type.DOUBLE),
    funPair("exp", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("expm1", Type.DOUBLE, "x" to Type.DOUBLE),
    funPair("floor", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("hypot", Type.DOUBLE, "x" to Type.DOUBLE, "y" to Type.DOUBLE),
    funPair("log", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("log10", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("max", Type.DOUBLE, "a" to Type.DOUBLE, "b" to Type.DOUBLE),
    funPair("min", Type.DOUBLE, "a" to Type.DOUBLE, "b" to Type.DOUBLE),
    funPair("pow", Type.DOUBLE, "a" to Type.DOUBLE, "b" to Type.INT),
    funPair("round", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("sin", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("sinh", Type.DOUBLE, "x" to Type.DOUBLE),
    funPair("sqrt", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("tan", Type.DOUBLE, "a" to Type.DOUBLE),
    funPair("tanh", Type.DOUBLE, "x" to Type.DOUBLE),
)

val allPredefinedFunctions = predefinedMathFunctions

private fun funPair(name: String, type: Type, vararg arguments: Pair<String, Type>) =
    name to Function(type, name, *arguments)