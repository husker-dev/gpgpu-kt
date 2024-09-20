package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type

interface Program {
    fun execute(instances: Int, vararg mapping: Pair<String, Source>)
    fun dealloc()
}

abstract class SimpleCProgram(ast: Scope): Program {
    protected val buffers = ast.fields.filter {
        Modifiers.IN in it.modifiers ||
        Modifiers.OUT in it.modifiers
    }.map { it.name }.toList()

    protected fun stringifyScope(scope: Scope, buffer: StringBuilder, ignoredFields: List<Field>? = null){
        scope.statements.forEach { statement ->
            stringifyStatement(statement, buffer, true, ignoredFields)
        }
    }

    protected open fun stringifyStatement(statement: Statement, buffer: StringBuilder, expressionSemicolon: Boolean, ignoredFields: List<Field>? = null){
        when(statement) {
            is ExpressionStatement -> {
                stringifyExpression(statement.expression, buffer)
                if(expressionSemicolon)
                    buffer.append(";")
            }
            is FunctionStatement -> stringifyFunction(statement.function, buffer)
            is FieldStatement -> stringifyFieldStatement(statement, buffer, ignoredFields)
            is ReturnStatement -> stringifyReturnStatement(statement, buffer)
            is IfStatement -> stringifyIfStatement(statement, buffer)
            is ForStatement -> stringifyForStatement(statement, buffer)
            is WhileStatement -> stringifyWhileStatement(statement, buffer)
            is EmptyStatement -> buffer.append(";")
            is BreakStatement -> buffer.append("break;")
            is ContinueStatement -> buffer.append("continue;")
        }
    }

    protected open fun stringifyWhileStatement(statement: WhileStatement, buffer: StringBuilder){
        buffer.append("while(")
        stringifyExpression(statement.condition, buffer)
        buffer.append("){")
        stringifyScope(statement.body, buffer)
        buffer.append("}")
    }

    protected open fun stringifyForStatement(statement: ForStatement, buffer: StringBuilder){
        buffer.append("for(")
        stringifyStatement(statement.initialization, buffer, true)
        stringifyStatement(statement.condition, buffer, true)
        stringifyStatement(statement.iteration, buffer, false)
        buffer.append("){")
        stringifyScope(statement.body, buffer)
        buffer.append("}")
    }

    protected open fun stringifyIfStatement(statement: IfStatement, buffer: StringBuilder){
        buffer.append("if(")
        stringifyExpression(statement.condition, buffer)
        buffer.append("){")
        stringifyScope(statement.body, buffer)
        buffer.append("}")
        if(statement.elseBody != null){
            buffer.append("else{")
            stringifyScope(statement.elseBody, buffer)
            buffer.append("}")
        }
    }

    protected open fun stringifyReturnStatement(returnStatement: ReturnStatement, buffer: StringBuilder){
        buffer.append("return")
        if(returnStatement.expression != null) {
            buffer.append(" ")
            stringifyExpression(returnStatement.expression, buffer)
        }
        buffer.append(";")
    }

    protected open fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder, ignoredFields: List<Field>?){
        val modifiers = fieldStatement.fields[0].modifiers
        val type = fieldStatement.fields[0].type

        if(Modifiers.IN in modifiers || Modifiers.OUT in modifiers)
            return

        if(modifiers.isNotEmpty())
            buffer.append(modifiers.joinToString(" ", postfix = " ") { it.text })
        buffer.append(type.toCType())
        buffer.append(" ")

        fieldStatement.fields.forEachIndexed { i, field ->
            buffer.append(field.name)
            if(field.initialExpression != null){
                buffer.append("=")
                stringifyExpression(field.initialExpression, buffer)
            }
            if(i == fieldStatement.fields.lastIndex)
                buffer.append(";")
            else buffer.append(",")
        }
    }

    protected open fun stringifyFunction(function: Function, buffer: StringBuilder, additionalModifier: String? = null){
        if(additionalModifier != null) {
            buffer.append(additionalModifier)
            buffer.append(" ")
        }
        stringifyModifiers(function.modifiers, buffer)
        buffer.append(function.returnType.text)
        buffer.append(" ")
        buffer.append(function.name)
        buffer.append("(")
        buffer.append(function.arguments.joinToString(",") {
            "${it.type.toCType()} ${it.name}"
        })
        buffer.append("){")
        stringifyScope(function, buffer, function.arguments)
        buffer.append("}")
    }

    protected open fun stringifyExpression(expression: Expression, buffer: StringBuilder){
        if(expression is AxBExpression){
            stringifyExpression(expression.left, buffer)
            buffer.append(expression.operator.token)
            stringifyExpression(expression.right, buffer)
        }
        if(expression is AxExpression){
            stringifyExpression(expression.left, buffer)
            buffer.append(expression.operator.token)
        }
        if(expression is XBExpression){
            buffer.append(expression.operator.token)
            stringifyExpression(expression.right, buffer)
        }
        if(expression is ArrayAccessExpression){
            buffer.append(expression.array.name)
            buffer.append("[")
            stringifyExpression(expression.index, buffer)
            buffer.append("]")
        }
        if(expression is FunctionCallExpression){
            buffer.append(expression.function.name)
            buffer.append("(")
            expression.arguments.forEachIndexed { i, arg ->
                stringifyExpression(arg, buffer)
                if(i != expression.arguments.lastIndex)
                    buffer.append(",")
            }
            buffer.append(")")
        }
        if(expression is FieldExpression)
            buffer.append(expression.field.name)
        if(expression is ConstExpression) {
            buffer.append(expression.lexeme.text)
            if(expression.type == Type.FLOAT)
                buffer.append("f")
        }
        if(expression is BracketExpression){
            buffer.append("(")
            stringifyExpression(expression.wrapped, buffer)
            buffer.append(")")
        }
        if(expression is CastExpression){
            buffer.append("(").append(expression.type.text).append(")")
            stringifyExpression(expression.right, buffer)
        }
    }

    protected open fun stringifyModifiers(modifiers: List<Modifiers>, buffer: StringBuilder){
        if(modifiers.isNotEmpty())
            buffer.append(modifiers.joinToString(" ", postfix = " ") { it.text })
    }
}

fun Type.toCType() = when(this){
    Type.VOID -> "void"
    Type.FLOAT -> "float"
    Type.INT -> "int"
    Type.BOOLEAN -> "bool"
    Type.FLOAT_ARRAY -> "float*"
    Type.INT_ARRAY -> "int*"
    Type.BOOLEAN_ARRAY -> "bool*"
}