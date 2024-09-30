package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.ast.lexer.Lexeme


class Function(
    val lexeme: Lexeme,
    val scope: Scope,
    val name: String,
    val modifiers: List<Modifiers>,
    val returnType: Type
){
    val arguments = mutableListOf<Field>()
    val argumentsTypes = mutableListOf<Type>()
    lateinit var body: ScopeStatement

    fun addArgument(argument: Field){
        arguments += argument
        argumentsTypes += argument.type
    }
}