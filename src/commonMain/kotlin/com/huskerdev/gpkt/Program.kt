package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFieldHeader
import com.huskerdev.gpkt.utils.appendCFunctionHeader

interface Program {
    fun execute(instances: Int, vararg mapping: Pair<String, Source>)
    fun dealloc()
}

abstract class SimpleCProgram(ast: Scope): Program {
    protected val buffers = ast.fields.filter {
        Modifiers.IN in it.modifiers ||
        Modifiers.OUT in it.modifiers
    }.map { it.name }.toList()

    protected fun stringifyScope(scope: Scope, buffer: StringBuilder){
        scope.statements.forEach { statement ->
            stringifyStatement(buffer, statement)
        }
    }

    protected open fun stringifyStatement(
        buffer: StringBuilder,
        statement: Statement
    ){
        when(statement) {
            is ExpressionStatement -> stringifyExpression(buffer, statement.expression, true)
            is FunctionStatement -> stringifyFunction(statement.function, buffer)
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

    protected open fun stringifyWhileStatement(statement: WhileStatement, buffer: StringBuilder){
        buffer.append("while(")
        stringifyExpression(buffer, statement.condition)
        buffer.append(")")
        if(statement.body.statements.size > 1) buffer.append("{")
        stringifyScope(statement.body, buffer)
        if(statement.body.statements.size > 1) buffer.append("}")
    }

    protected open fun stringifyForStatement(statement: ForStatement, buffer: StringBuilder){
        buffer.append("for(")
        stringifyStatement(buffer, statement.initialization)
        if(statement.condition != null) stringifyExpression(buffer, statement.condition)
        buffer.append(";")
        if(statement.iteration != null) stringifyExpression(buffer, statement.iteration)
        buffer.append(")")
        if(statement.body.statements.size > 1) buffer.append("{")
        stringifyScope(statement.body, buffer)
        if(statement.body.statements.size > 1) buffer.append("}")
    }

    protected open fun stringifyIfStatement(statement: IfStatement, buffer: StringBuilder){
        buffer.append("if(")
        stringifyExpression(buffer, statement.condition)
        buffer.append(")")
        if(statement.body.statements.size > 1) buffer.append("{")
        stringifyScope(statement.body, buffer)
        if(statement.body.statements.size > 1) buffer.append("}")
        if(statement.elseBody != null){
            buffer.append("else")
            if(statement.elseBody.statements.size > 1) buffer.append("{")
            stringifyScope(statement.elseBody, buffer)
            if(statement.elseBody.statements.size > 1) buffer.append("}")
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
        val modifiers = fieldStatement.fields[0].modifiers
        val type = fieldStatement.fields[0].type
        if(Modifiers.IN in modifiers || Modifiers.OUT in modifiers)
            return

        appendCFieldHeader(
            buffer = buffer,
            modifiers = modifiers.map { it.text },
            type = type.toCType(fieldStatement.scope.parentScope == null),
            fields = fieldStatement.fields,
            expressionGen = { stringifyExpression(buffer, it) }
        )
    }

    protected open fun stringifyFunction(function: Function, buffer: StringBuilder){
        appendCFunctionHeader(
            buffer = buffer,
            modifiers = function.modifiers.map { it.text },
            type = function.returnType.toCType(false),
            name = function.name,
            args = function.arguments.map { "${it.type.toCType(false)} ${it.name}" }
        )
        stringifyScope(function, buffer)
        buffer.append("}")
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
        if(expression.type == Type.FLOAT)
            buffer.append("f")
    }

    protected open fun stringifyBracketExpression(buffer: StringBuilder, expression: BracketExpression){
        buffer.append("(")
        stringifyExpression(buffer, expression.wrapped)
        buffer.append(")")
    }

    protected open fun stringifyCastExpression(buffer: StringBuilder, expression: CastExpression){
        buffer.append("(").append(expression.type.toCType(false)).append(")")
        stringifyExpression(buffer, expression.right)
    }

    protected open fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression){
        buffer.append(expression.field.name)
    }

    protected open fun Type.toCType(isGlobal: Boolean) = when(this){
        Type.VOID -> "void"
        Type.FLOAT -> "float"
        Type.INT -> "int"
        Type.BOOLEAN -> "bool"
        Type.FLOAT_ARRAY -> "float*"
        Type.INT_ARRAY -> "int*"
        Type.BOOLEAN_ARRAY -> "bool*"
    }
}

