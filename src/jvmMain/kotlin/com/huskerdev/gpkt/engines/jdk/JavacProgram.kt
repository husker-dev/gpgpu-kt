package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.Program
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.engines.cpu.AbstractThread
import com.huskerdev.gpkt.engines.cpu.runThread
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Method
import java.util.concurrent.atomic.AtomicLong


class JavacProgram(ast: Scope): Program {

    companion object {
        val counter = AtomicLong()
    }

    private val buffers = ast.fields.filter {
        Modifiers.IN in it.modifiers ||
        Modifiers.OUT in it.modifiers
    }.map { it.name }.toList()

    private val execMethod: Method

    init {
        val className = "GPJavacProgram${counter.getAndIncrement()}"
        val buffer = StringBuilder()
        buffer.append("""
            public class $className{ 
                public static void _execute(int fromIndex, int toIndex, ${buffers.joinToString { "float[] $it" }}){
                    for(int i = fromIndex; i < toIndex; i++)
                        main(${(arrayOf("i") + buffers).joinToString(",")});
                }
                
        """.trimIndent())
        stringifyScope(ast, buffer)
        buffer.append("}")

        val clazz = ClassCompiler.compileClass(buffer.toString(), className)
        execMethod = clazz.getMethod(
            "_execute",
            *(arrayListOf(Int::class.java, Int::class.java) + buffers.map { FloatArray::class.java }).toTypedArray()
        )
    }

    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        val map = hashMapOf(*mapping)

        val arrays = buffers.map {
            if(it !in map) throw Exception("Source '$it' have not been set")
            else (map[it] as JavacSource).array
        }

        val threads = Runtime.getRuntime().availableProcessors()

        if(threads == 1){
            execPeriod(arrays, 0, instances)
        } else if(instances > threads) {
            val instancesPerThread = instances / threads
            val threadList = arrayListOf<AbstractThread>()
            for (i in 0 until threads) {
                val fromIndex = i * instancesPerThread
                threadList += runThread {
                    execPeriod(arrays, fromIndex, fromIndex + instancesPerThread)
                }
            }
            threadList.forEach { it.waitEnd() }
        }else {
            val threadList = arrayListOf<AbstractThread>()
            for(i in 0 until instances){
                threadList += runThread {
                    execPeriod(arrays, i, i+1)
                }
            }
            threadList.forEach { it.waitEnd() }
        }
    }

    private fun execPeriod(arrays: List<FloatArray>, from: Int, to: Int) {
        try {
            execMethod.invoke(null, *(listOf(from, to) + arrays).toTypedArray())
        }catch (e: InvocationTargetException){
            throw e.targetException
        }
    }


    override fun dealloc() = Unit


    // Stringify

    private fun stringifyScope(scope: Scope, buffer: StringBuilder, ignoredFields: List<Field>? = null){
        scope.statements.forEach { statement ->
            stringifyStatement(statement, buffer, true, ignoredFields)
        }
    }

    private fun stringifyStatement(
        statement: Statement,
        buffer: StringBuilder,
        expressionSemicolon: Boolean,
        ignoredFields: List<Field>? = null
    ){
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

    private fun stringifyWhileStatement(statement: WhileStatement, buffer: StringBuilder){
        buffer.append("while(")
        stringifyExpression(statement.condition, buffer)
        buffer.append("){")
        stringifyScope(statement.body, buffer)
        buffer.append("}")
    }

    private fun stringifyForStatement(statement: ForStatement, buffer: StringBuilder){
        buffer.append("for(")
        stringifyStatement(statement.initialization, buffer, true)
        stringifyStatement(statement.condition, buffer, true)
        stringifyStatement(statement.iteration, buffer, false)
        buffer.append("){")
        stringifyScope(statement.body, buffer)
        buffer.append("}")
    }

    private fun stringifyIfStatement(statement: IfStatement, buffer: StringBuilder){
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

    private fun stringifyReturnStatement(returnStatement: ReturnStatement, buffer: StringBuilder){
        buffer.append("return")
        if(returnStatement.expression != null) {
            buffer.append(" ")
            stringifyExpression(returnStatement.expression, buffer)
        }
        buffer.append(";")
    }

    private fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder, ignoredFields: List<Field>?){
        val modifiers = fieldStatement.fields[0].modifiers
        val type = fieldStatement.fields[0].type

        if(Modifiers.IN in modifiers || Modifiers.OUT in modifiers)
            return

        if(fieldStatement.scope.parentScope == null)
            buffer.append("private static ")
        if(modifiers.isNotEmpty())
            buffer.append(modifiers.joinToString(" ", postfix = " ") { it.text })
        buffer.append(type.toJavaType())
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

    private fun stringifyFunction(function: Function, buffer: StringBuilder, additionalModifier: String? = null){
        if(additionalModifier != null) {
            buffer.append(additionalModifier)
            buffer.append(" ")
        }
        buffer.append("private static ")
        stringifyModifiers(function.modifiers, buffer)
        buffer.append(function.returnType.text)
        buffer.append(" ")
        buffer.append(function.name)
        buffer.append("(")

        if(function.name == "main"){
            buffer.append((
                function.arguments.map { "${it.type.toJavaType()} ${it.name}" } +
                buffers.map { "float[] $it" }
            ).joinToString(","))
        }else {
            buffer.append(function.arguments.joinToString(",") {
                "${it.type.toJavaType()} ${it.name}"
            })
        }
        buffer.append("){")
        stringifyScope(function, buffer, function.arguments)
        buffer.append("}")
    }

    private fun stringifyExpression(expression: Expression, buffer: StringBuilder){
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

    private fun stringifyModifiers(modifiers: List<Modifiers>, buffer: StringBuilder){
        if(modifiers.isNotEmpty())
            buffer.append(modifiers.joinToString(" ", postfix = " ") { it.text })
    }
}

fun Type.toJavaType() = when(this){
    Type.VOID -> "void"
    Type.FLOAT -> "float"
    Type.INT -> "int"
    Type.BOOLEAN -> "bool"
    Type.FLOAT_ARRAY -> "float[]"
    Type.INT_ARRAY -> "int[]"
    Type.BOOLEAN_ARRAY -> "bool[]"
}