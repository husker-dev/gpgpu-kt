package com.huskerdev.gpkt.utils

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.*
import com.huskerdev.gpkt.ast.types.*


abstract class SimpleCProgram(
    ast: ScopeStatement,
    private val useStruct: Boolean = true,
    private val useCArrayDefs: Boolean = true,
    private val useFunctionDefs: Boolean = true
): GPProgram(ast) {

    abstract fun stringifyMainFunctionDefinition(buffer: StringBuilder, function: GPFunction)
    abstract fun stringifyMainFunctionBody(buffer: StringBuilder, function: GPFunction)
    abstract fun stringifyModifiersInStruct(field: GPField): String
    abstract fun stringifyModifiersInGlobal(obj: Any): String
    abstract fun stringifyModifiersInLocal(field: GPField): String
    abstract fun stringifyModifiersInArg(field: GPField): String

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
            is ArrayDefinitionExpression -> stringifyArrayDefinitionExpression(buffer, expression)
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
        if(statement.scope.parentScope == null) {
            if (useStruct) {
                buffer.append("typedef struct{")
                fun stringify(field: GPField) {
                    val modifiers = stringifyModifiersInStruct(field)
                    if (modifiers.isNotEmpty())
                        buffer.append(modifiers).append(" ")

                    buffer.append(toCType(field.type))
                        .append(" ")
                        .append(
                            if (field.type is ArrayPrimitiveType<*>)
                                toCArrayName(field.obfName, field.type.size)
                            else field.obfName
                        )
                    buffer.append(";")
                }
                buffers.forEach(::stringify)
                locals.forEach(::stringify)
                buffer.append("}__in;")
            } else {
                statement.statements.filter {
                    it is FieldStatement && (
                            Modifiers.EXTERNAL in it.fields[0].modifiers ||
                            Modifiers.THREADLOCAL in it.fields[0].modifiers)
                }.forEach {
                    stringifyFieldStatement(it as FieldStatement, buffer, true)
                }
            }
        }

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
        // If type is fixed size array
        if(useCArrayDefs &&
            returnStatement.expression != null &&
            returnStatement.expression.type.isArray &&
            !returnStatement.expression.type.isDynamicArray
        ){
            val expr = returnStatement.expression
            val type = expr.type
            val size = (type as ArrayPrimitiveType<*>).size

            when (expr) {
                // If create array
                is ArrayDefinitionExpression -> {
                    buffer.append("{")
                    for (i in 0 until size) {
                        buffer.append("__ret[").append(i).append("]=")
                        stringifyExpression(buffer, expr.elements[i], true)
                    }
                    buffer.append("return;")
                    buffer.append("}")
                    return
                }
                // If return array type
                is FieldExpression -> {
                    buffer.append("{")
                    for (i in 0 until size)
                        buffer.append("__ret[").append(i).append("]=").append(expr.field.obfName).append("[").append(i).append("];")
                    buffer.append("return;")
                    buffer.append("}")
                    return
                }
                else -> throw UnsupportedOperationException()
            }
        }

        buffer.append("return")
        if(returnStatement.expression != null) {
            buffer.append(" ")
            stringifyExpression(buffer, returnStatement.expression)
        }
        buffer.append(";")
    }

    protected open fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder, force: Boolean = false){
        val fields = fieldStatement.fields
        val modifiers = fields[0].modifiers
        if(!force && (Modifiers.EXTERNAL in modifiers || Modifiers.THREADLOCAL in modifiers))
            return

        val type = fields[0].type

        val modifiersText = if(fieldStatement.scope.parentScope == null)
            stringifyModifiersInGlobal(fieldStatement.fields[0])
        else stringifyModifiersInLocal(fieldStatement.fields[0])
        if(modifiersText.isNotEmpty())
            buffer.append(modifiersText).append(" ")

        buffer.append(toCType(type)).append(" ")
        fields.forEachIndexed { i, field ->
            if(type is ArrayPrimitiveType<*>)
                buffer.append(toCArrayName(field.obfName, type.size))
            else
                buffer.append(field.obfName)
            if(field.initialExpression != null){

                // If fixed size array and initialized via function
                if(useCArrayDefs &&
                    type is ArrayPrimitiveType<*> &&
                    !type.isDynamicArray &&
                    field.initialExpression is FunctionCallExpression
                ){
                    buffer.append(";")
                    stringifyFunctionCallExpression(buffer, field.initialExpression as FunctionCallExpression, field)
                }else {
                    buffer.append("=")
                    val needCast = type != field.initialExpression!!.type
                    if (needCast)
                        buffer.append("(").append(toCType(type)).append(")(")
                    stringifyExpression(buffer, field.initialExpression!!)
                    if (needCast)
                        buffer.append(")")
                }
            }
            if(i != fields.lastIndex)
                buffer.append(",")
        }
        buffer.append(";")
    }

    protected open fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder){
        val function = statement.function
        if(function.name == "main"){
            stringifyMainFunctionDefinition(buffer, function)
            buffer.append("{")

            // Inputs struct
            if(useStruct) {
                buffer.append("__in __v={")
                buffers.forEachIndexed { index, field ->
                    buffer.append("__v").append(field.obfName)
                    if (index != buffers.lastIndex || locals.isNotEmpty())
                        buffer.append(",")
                }
                locals.forEachIndexed { index, field ->
                    stringifyExpression(buffer, field.initialExpression!!, false)
                    if (index != buffers.lastIndex)
                        buffer.append(",")
                }
                buffer.append("};")
            }
            stringifyMainFunctionBody(buffer, function)
            stringifyScopeStatement(buffer, statement.function.body!!, false)
            buffer.append("}")
        }else {
            if(statement is FunctionDefinitionStatement && !useFunctionDefs)
                return

            val modifiers = stringifyModifiersInGlobal(function)
            if(modifiers.isNotEmpty())
                buffer.append(modifiers).append(" ")

            val args = function.arguments.map(::convertToFuncArg).toMutableList()
            if(useStruct)
                args.add(0, "__in __v")

            // If type is fixed size array
            val type = function.returnType
            if(useCArrayDefs && type is ArrayPrimitiveType<*> && !type.isDynamicArray){
                val singleType = type.single as SinglePrimitiveType<*>
                args.add(0, "${toCType(singleType.toDynamicArray())} ${toCArrayName("__ret", -1)}")

                appendCFunctionDefinition(
                    buffer = buffer,
                    type = toCType(VOID),
                    name = function.obfName,
                    args = args
                )
                stringifyStatement(buffer, statement.function.body!!)

                // Single-element access function
                if(modifiers.isNotEmpty())
                    buffer.append(modifiers).append(" ")

                args[0] = "${toCType(INT)} __i"
                appendCFunctionDefinition(
                    buffer = buffer,
                    type = toCType(singleType),
                    name = "__${function.obfName}",
                    args = args
                )
                buffer.append("{")
                    .append(toCType(type))
                    .append(" ")
                    .append(toCArrayName("t", type.size))
                    .append(";")
                    .append(function.obfName)
                    .append("(t")
                if(args.size > 1) buffer.append(",")
                if(useStruct) {
                    buffer.append("__v")
                    if (args.size > 2) buffer.append(",")
                }
                function.arguments.forEachIndexed { i, arg ->
                    buffer.append(arg.obfName)
                    if(i < function.arguments.size - 1)
                        buffer.append(",")
                }
                buffer.append(");return t[__i];}")
            }else {
                appendCFunctionDefinition(
                    buffer = buffer,
                    type = toCType(function.returnType),
                    name = if (function.returnType is ArrayPrimitiveType<*>)
                        toCArrayName(function.obfName, function.returnType.size) else function.obfName,
                    args = args
                )
                if(statement !is FunctionDefinitionStatement)
                    stringifyStatement(buffer, statement.function.body!!)
                else
                    buffer.append(";")
            }
        }
    }

    /* ================== *\
           Expressions
    \* ================== */

    protected open fun stringifyArrayDefinitionExpression(buffer: StringBuilder, expression: ArrayDefinitionExpression){
        buffer.append("{")
        expression.elements.forEachIndexed { i, e ->
            stringifyExpression(buffer, e)
            if(i < expression.elements.size - 1)
                buffer.append(",")
        }
        buffer.append("}")
    }

    protected open fun stringifyAxBExpression(buffer: StringBuilder, expression: AxBExpression){
        // If assign to function that returns fixed-size array
        if(useCArrayDefs &&
            expression.operator == Operator.ASSIGN &&
            expression.left is FieldExpression &&
            expression.left.type.isArray &&
            !expression.left.type.isDynamicArray &&
            expression.right is FunctionCallExpression
        ){
            stringifyFunctionCallExpression(buffer, expression.right, expression.left.field)
        }else {
            stringifyExpression(buffer, expression.left)
            buffer.append(expression.operator.token)
            stringifyExpression(buffer, expression.right)
        }
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
        // If array is a function
        if(useCArrayDefs && expression.array is FunctionCallExpression){
            stringifyFunctionCallExpression(buffer, expression.array, arrayIndex = expression.index)
        }else {
            stringifyExpression(buffer, expression.array)
            buffer.append("[")
            stringifyExpression(buffer, expression.index)
            buffer.append("]")
        }
    }

    protected open fun stringifyFunctionCallExpression(
        buffer: StringBuilder,
        expression: FunctionCallExpression,
        arrayField: GPField? = null,
        arrayIndex: Expression? = null
    ){
        val function = expression.function
        val isPredefined = function.obfName in predefinedMathFunctions

        if(isPredefined)
            buffer.append("(").append(toCType(function.returnType)).append(")")
        if(arrayIndex != null)
            buffer.append("__")
        buffer.append(if(isPredefined) convertPredefinedFunctionName(expression) else function.obfName)
        buffer.append("(")

        val passContextVars = useStruct && !isPredefined

        if(arrayField != null){
            buffer.append(arrayField.obfName)
            if (expression.arguments.isNotEmpty() || passContextVars)
                buffer.append(",")
        }
        if(arrayIndex != null){
            stringifyExpression(buffer, arrayIndex)
            if (expression.arguments.isNotEmpty() || passContextVars)
                buffer.append(",")
        }
        if(passContextVars) {
            buffer.append("__v")
            if (expression.arguments.isNotEmpty())
                buffer.append(",")
        }
        expression.arguments.forEachIndexed { i, arg ->
            val needCast = arg.type != function.argumentsTypes[i]
            if(isPredefined && needCast)
                buffer.append("(").append(toCType(function.argumentsTypes[i])).append(")(")
            stringifyExpression(buffer, arg)
            if(needCast)
                buffer.append(")")
            if(i != expression.arguments.lastIndex)
                buffer.append(",")
        }
        buffer.append(")")
    }

    open fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) = functionExpression.function.obfName
    open fun convertPredefinedFieldName(field: GPField) = field.obfName

    protected open fun stringifyConstExpression(buffer: StringBuilder, expression: ConstExpression){
        buffer.append(expression.lexeme.text)
        if(expression.type.isFloating && "." !in expression.lexeme.text)
            buffer.append(".0")
        if(expression.type == FLOAT)
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
        if(useStruct && (expression.field.isExtern || expression.field.isLocal))
            buffer.append("__v.")
        val name = expression.field.obfName
        buffer.append(if(name in allPredefinedFields) convertPredefinedFieldName(expression.field) else name)
    }

    protected open fun convertToFuncArg(field: GPField): String{
        val buffer = StringBuilder()
        val modifiers = stringifyModifiersInArg(field)
        if(modifiers.isNotEmpty())
            buffer.append(modifiers).append(" ")

        buffer.append(toCType(field.type)).append(" ")
        if(field.type is ArrayPrimitiveType<*>)
            buffer.append(toCArrayName(field.obfName, field.type.size))
        else buffer.append(field.obfName)
        return buffer.toString()
    }

    protected open fun toCType(type: PrimitiveType) = when(type) {
        is VoidType -> "void"
        is FloatType, is FloatArrayType -> "float"
        is IntType, is IntArrayType -> "int"
        is ByteType, is ByteArrayType -> "char"
        is BooleanType -> "bool"
        else -> throw UnsupportedOperationException()
    }

    protected open fun toCArrayName(name: String, size: Int) =
        if(size == -1) "*$name" else "$name[$size]"
}

fun appendCFunctionDefinition(
    buffer: StringBuilder,
    type: String,
    name: String,
    args: List<String>
){
    buffer.append(type).append(" ").append(name).append("(")
    args.joinTo(buffer, separator = ",")
    buffer.append(")")
}