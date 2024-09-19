package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.ast.lexer.Lexeme


class Function(
    val lexeme: Lexeme,
    val scope: Scope,
    val name: String,
    val modifiers: List<Modifiers>,
    returnType: Type
): Scope(scope, returnType){
    val arguments = mutableListOf<Field>()
    val argumentsTypes = mutableListOf<Type>()

    override val returnType = super.returnType!!

    fun addArgument(argument: Field, lexeme: Lexeme, codeBlock: String){
        arguments += argument
        argumentsTypes += argument.type
        addField(argument, lexeme, codeBlock)
    }
}