package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFunctionHeader

interface Program {
    fun executeRange(
        indexOffset: Int,
        instances: Int,
        vararg mapping: Pair<String, Any>
    )

    fun execute(
        instances: Int,
        vararg mapping: Pair<String, Any>
    ) = executeRange(0, instances, *mapping)

    fun dealloc()
}

class FieldNotSetException(name: String): Exception("Field '$name' have not been set")

class TypesMismatchException(argument: String): Exception("Value type for argument '$argument' doesn't match.")

abstract class BasicProgram(ast: ScopeStatement): Program {
    protected val buffers = ast.scope.fields.filter {
        Modifiers.EXTERNAL in it.modifiers
    }.toList()

    fun areEqualTypes(actual: Any, expected: Type): Boolean{
        val actualType = when(actual){
            is AsyncFloatMemoryPointer, is SyncFloatMemoryPointer -> Type.FLOAT_ARRAY
            is AsyncIntMemoryPointer, is SyncIntMemoryPointer -> Type.INT_ARRAY
            is AsyncByteMemoryPointer, is SyncByteMemoryPointer -> Type.BYTE_ARRAY
            is Float -> Type.FLOAT
            is Int -> Type.INT
            is Byte -> Type.BYTE
            else -> throw UnsupportedOperationException("Unsupported type: '${actual::class}'")
        }
        return actualType == expected
    }
}

abstract class SimpleCProgram(ast: ScopeStatement): BasicProgram(ast) {

    protected open fun stringifyStatement(
        buffer: StringBuilder,
        statement: Statement
    ){
        when(statement) {
            is ScopeStatement -> stringifyScopeStatement(buffer, statement, true)
            is ExpressionStatement -> stringifyExpression(buffer, statement.expression, true)
            is FunctionStatement -> stringifyFunctionStatement(statement, buffer)
            is FieldStatement -> stringifyFieldStatement(statement, buffer)
            is ReturnStatement -> stringifyReturnStatement(statement, buffer)
            is IfStatement -> stringifyIfStatement(statement, buffer)
            is ForStatement -> stringifyForStatement(statement, buffer)
            is WhileStatement -> stringifyWhileStatement(statement, buffer)
            is EmptyStatement -> buffer.append(";")
            is BreakStatement -> buffer.append("break;")
            is ContinueStatement -> buffer.append("continue;")
        }
    }

    protected open fun stringifyExpression(
        buffer: StringBuilder,
        expression: Expression,
        semicolon: Boolean = false
    ){
        when(expression){
            is AxBExpression -> stringifyAxBExpression(buffer, expression)
            is AxExpression -> stringifyAxExpression(buffer, expression)
            is XBExpression -> stringifyXBExpression(buffer, expression)
            is ArrayAccessExpression -> stringifyArrayAccessExpression(buffer, expression)
            is FunctionCallExpression -> stringifyFunctionCallExpression(buffer, expression)
            is ConstExpression -> stringifyConstExpression(buffer, expression)
            is BracketExpression -> stringifyBracketExpression(buffer, expression)
            is CastExpression -> stringifyCastExpression(buffer, expression)
            is FieldExpression -> stringifyFieldExpression(buffer, expression)
        }
        if(semicolon)
            buffer.append(";")
    }

    /* ================== *\
           Statements
    \* ================== */

    protected fun stringifyScopeStatement(
        buffer: StringBuilder,
        statement: ScopeStatement,
        brackets: Boolean
    ){
        if(brackets) buffer.append("{")
        statement.statements.forEach { st ->
            stringifyStatement(buffer, st)
        }
        if(brackets) buffer.append("}")
    }

    protected open fun stringifyWhileStatement(statement: WhileStatement, buffer: StringBuilder){
        buffer.append("while(")
        stringifyExpression(buffer, statement.condition)
        buffer.append(")")
        stringifyStatement(buffer, statement.body)
    }

    protected open fun stringifyForStatement(statement: ForStatement, buffer: StringBuilder){
        buffer.append("for(")
        stringifyStatement(buffer, statement.initialization)
        if(statement.condition != null) stringifyExpression(buffer, statement.condition)
        buffer.append(";")
        if(statement.iteration != null) stringifyExpression(buffer, statement.iteration)
        buffer.append(")")
        stringifyStatement(buffer, statement.body)
    }

    protected open fun stringifyIfStatement(statement: IfStatement, buffer: StringBuilder){
        buffer.append("if(")
        stringifyExpression(buffer, statement.condition)
        buffer.append(")")
        stringifyStatement(buffer, statement.body)
        if(statement.elseBody != null){
            buffer.append("else ")
            stringifyStatement(buffer, statement.elseBody)
        }
    }

    protected open fun stringifyReturnStatement(returnStatement: ReturnStatement, buffer: StringBuilder){
        buffer.append("return")
        if(returnStatement.expression != null) {
            buffer.append(" ")
            stringifyExpression(buffer, returnStatement.expression)
        }
        buffer.append(";")
    }

    protected open fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder){
        val fields = fieldStatement.fields
        val modifiers = fields[0].modifiers
        val type = fields[0].type
        modifiers.joinTo(buffer, separator = " ", transform = ::toCModifier)
        if(modifiers.isNotEmpty())
            buffer.append(" ")

        buffer.append(toCType(type)).append(" ")
        fields.forEachIndexed { i, field ->
            if(type.isArray)
                buffer.append(toCArrayName(field.name))
            else
                buffer.append(field.name)
            if(field.initialExpression != null){
                buffer.append("=")
                stringifyExpression(buffer, field.initialExpression)
            }
            if(i != fields.lastIndex)
                buffer.append(",")
        }
        buffer.append(";")
    }

    protected open fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder){
        val function = statement.function
        appendCFunctionHeader(
            buffer = buffer,
            modifiers = function.modifiers.map { it.text },
            type = convertToReturnType(function.returnType),
            name = function.name,
            args = function.arguments.map(::convertToFuncArg)
        )
        stringifyStatement(buffer, statement.function.body)
    }

    /* ================== *\
           Expressions
    \* ================== */

    protected open fun stringifyAxBExpression(buffer: StringBuilder, expression: AxBExpression){
        stringifyExpression(buffer, expression.left)
        buffer.append(expression.operator.token)
        stringifyExpression(buffer, expression.right)
    }

    protected open fun stringifyAxExpression(buffer: StringBuilder, expression: AxExpression){
        stringifyExpression(buffer, expression.left)
        buffer.append(expression.operator.token)
    }

    protected open fun stringifyXBExpression(buffer: StringBuilder, expression: XBExpression){
        buffer.append(expression.operator.token)
        stringifyExpression(buffer, expression.right)
    }

    protected open fun stringifyArrayAccessExpression(buffer: StringBuilder, expression: ArrayAccessExpression){
        buffer.append(expression.array.name)
        buffer.append("[")
        stringifyExpression(buffer, expression.index)
        buffer.append("]")
    }

    protected open fun stringifyFunctionCallExpression(buffer: StringBuilder, expression: FunctionCallExpression){
        buffer.append(expression.function.name)
        buffer.append("(")
        expression.arguments.forEachIndexed { i, arg ->
            stringifyExpression(buffer, arg)
            if(i != expression.arguments.lastIndex)
                buffer.append(",")
        }
        buffer.append(")")
    }

    protected open fun stringifyConstExpression(buffer: StringBuilder, expression: ConstExpression){
        buffer.append(expression.lexeme.text)
        if(expression.type.isFloating && "." !in expression.lexeme.text)
            buffer.append(".0")
        if(expression.type == Type.FLOAT)
            buffer.append("f")
    }

    protected open fun stringifyBracketExpression(buffer: StringBuilder, expression: BracketExpression){
        buffer.append("(")
        stringifyExpression(buffer, expression.wrapped)
        buffer.append(")")
    }

    protected open fun stringifyCastExpression(buffer: StringBuilder, expression: CastExpression){
        buffer.append("(").append(toCType(expression.type)).append(")")
        stringifyExpression(buffer, expression.right)
    }

    protected open fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression){
        buffer.append(expression.field.name)
    }

    protected open fun convertToFuncArg(field: Field): String{
        val buffer = StringBuilder()
        field.modifiers.joinTo(buffer, separator = " ", transform = ::toCModifier)
        if(field.modifiers.isNotEmpty())
            buffer.append(" ")

        buffer.append(toCType(field.type)).append(" ")
        if(field.type.isArray)
            buffer.append(toCArrayName(field.name))
        else buffer.append(field.name)
        return buffer.toString()
    }

    protected open fun convertToReturnType(type: Type) =
        if(type.isArray) "${toCType(type)}*"
        else toCType(type)

    protected open fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.EXTERNAL -> "extern"
        Modifiers.CONST -> "const"
        Modifiers.READONLY -> ""
    }

    protected open fun toCType(type: Type) = when(type) {
        Type.VOID -> "void"
        Type.FLOAT, Type.FLOAT_ARRAY -> "float"
        Type.INT, Type.INT_ARRAY -> "int"
        Type.BYTE, Type.BYTE_ARRAY -> "char"
        Type.BOOLEAN, Type.BOOLEAN_ARRAY -> "bool"
    }

    protected open fun toCArrayName(name: String) = "*$name"
}

