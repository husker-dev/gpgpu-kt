package com.huskerdev.gpkt.utils

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.*
import com.huskerdev.gpkt.ast.types.*


abstract class CProgramPrinter(
    protected val ast: GPScope,
    protected val buffers: List<GPField>,
    protected val locals: List<GPField>,

    // Printer settings
    private val useExternC: Boolean = false,
    private val useLocalStruct: Boolean = true,
    private val useArrayStruct: Boolean = true,
    private val useArrayStructCast: Boolean = true, // C++ struct creation style (active when 'useArrayStruct' is true)
    private val useFunctionDefs: Boolean = true,
    private val useStructClasses: Boolean = true,
) {
    companion object {
        var debug = false
    }

    private var contextClass: GPClass? = null

    open fun stringify(): String{
        val buffer = StringBuilder()

        val header = hashMapOf<String, String>()
        stringifyScope(header, buffer, ast, false)

        val headerBuffer = StringBuilder()
        header.map { it.value }.joinTo(headerBuffer)
        buffer.insert(0, headerBuffer)

        if(useExternC) {
            buffer.insert(0, "extern \"C\"{")
            buffer.append("}")
        }
        if(debug)
            println(buffer)
        return buffer.toString()
    }

    abstract fun stringifyMainFunctionDefinition(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction)
    abstract fun stringifyMainFunctionBody(header: MutableMap<String, String>, buffer: StringBuilder, function: GPFunction)
    abstract fun stringifyModifiersInStruct(field: GPField): String
    abstract fun stringifyModifiersInGlobal(obj: Any): String
    abstract fun stringifyModifiersInLocal(field: GPField): String
    abstract fun stringifyModifiersInArg(field: GPField): String
    abstract fun stringifyModifiersInLocalsStruct(): String

    protected open fun stringifyStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        statement: Statement
    ){
        when(statement) {
            is ScopeStatement -> stringifyScopeStatement(header, buffer, statement, true)
            is ExpressionStatement -> stringifyExpression(header, buffer, statement.expression, true)
            is FunctionStatement -> stringifyFunctionStatement(header, buffer, statement)
            is FieldStatement -> stringifyFieldStatement(header, buffer, statement)
            is ReturnStatement -> stringifyReturnStatement(header, buffer, statement)
            is IfStatement -> stringifyIfStatement(header, buffer, statement)
            is ForStatement -> stringifyForStatement(header, buffer, statement)
            is WhileStatement -> stringifyWhileStatement(header, buffer, statement)
            is ClassStatement -> stringifyClassStatement(header, buffer, statement)
            is EmptyStatement -> buffer.append(";")
            is BreakStatement -> buffer.append("break;")
            is ContinueStatement -> buffer.append("continue;")
        }
    }

    protected open fun stringifyExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: Expression,
        semicolon: Boolean = false
    ){
        when(expression){
            is ArrayDefinitionExpression -> stringifyArrayDefinitionExpression(header, buffer, expression)
            is AxBExpression -> stringifyAxBExpression(header, buffer, expression)
            is AxExpression -> stringifyAxExpression(header, buffer, expression)
            is XBExpression -> stringifyXBExpression(header, buffer, expression)
            is ArrayAccessExpression -> stringifyArrayAccessExpression(header, buffer, expression)
            is FunctionCallExpression -> stringifyFunctionCallExpression(header, buffer, expression)
            is ConstExpression -> stringifyConstExpression(header, buffer, expression)
            is BracketExpression -> stringifyBracketExpression(header, buffer, expression)
            is CastExpression -> stringifyCastExpression(header, buffer, expression)
            is FieldExpression -> stringifyFieldExpression(header, buffer, expression)
            is ClassCreationExpression -> stringifyClassCreationExpression(header, buffer, expression)
        }
        if(semicolon)
            buffer.append(";")
    }

    /* ================== *\
           Statements
    \* ================== */

    protected fun stringifyScopeStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        statement: ScopeStatement,
        brackets: Boolean
    ) = stringifyScope(header, buffer, statement.scopeObj, brackets)


    protected fun stringifyScope(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        scope: GPScope,
        brackets: Boolean
    ){
        if(scope.parentScope == null) {
            if (useLocalStruct) {
                buffer.append("typedef struct{")
                fun stringify(field: GPField) {
                    val modifiers = stringifyModifiersInStruct(field)
                    if (modifiers.isNotEmpty())
                        buffer.append(modifiers).append(" ")

                    buffer.append(toCType(header, field.type))
                        .append(" ")
                        .append(
                            if (field.type is ArrayPrimitiveType<*>)
                                convertArrayName(field.obfName, field.type.size)
                            else field.obfName
                        )
                    buffer.append(";")
                }
                buffers.forEach(::stringify)
                locals.forEach(::stringify)
                buffer.append("}__in;")
            } else {
                scope.statements.filter {
                    it is FieldStatement && (
                            Modifiers.EXTERNAL in it.fields[0].modifiers ||
                                    Modifiers.THREADLOCAL in it.fields[0].modifiers)
                }.forEach {
                    stringifyFieldStatement(header, buffer, it as FieldStatement, true)
                }
            }
        }

        if(brackets) buffer.append("{")
        scope.statements.forEach { st ->
            stringifyStatement(header, buffer, st)
        }
        if(brackets) buffer.append("}")
    }

    protected open fun stringifyWhileStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        statement: WhileStatement
    ){
        buffer.append("while(")
        stringifyExpression(header, buffer, statement.condition)
        buffer.append(")")
        stringifyStatement(header, buffer, statement.body)
    }

    protected open fun stringifyForStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        statement: ForStatement
    ){
        buffer.append("for(")
        stringifyStatement(header, buffer, statement.initialization)
        if(statement.condition != null) stringifyExpression(header, buffer, statement.condition)
        buffer.append(";")
        if(statement.iteration != null) stringifyExpression(header, buffer, statement.iteration)
        buffer.append(")")
        stringifyStatement(header, buffer, statement.body)
    }

    protected open fun stringifyIfStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        statement: IfStatement
    ){
        buffer.append("if(")
        stringifyExpression(header, buffer, statement.condition)
        buffer.append(")")
        stringifyStatement(header, buffer, statement.body)
        if(statement.elseBody != null){
            buffer.append("else ")
            stringifyStatement(header, buffer, statement.elseBody)
        }
    }

    protected open fun stringifyReturnStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        returnStatement: ReturnStatement
    ){
        buffer.append("return")
        if(returnStatement.expression != null) {
            buffer.append(" ")
            stringifyExpression(header, buffer, returnStatement.expression)
        }
        buffer.append(";")
    }

    protected open fun stringifyFieldStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        fieldStatement: FieldStatement,
        force: Boolean = false
    ){
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

        buffer.append(toCType(header, type)).append(" ")
        fields.forEachIndexed { i, field ->
            if(type is ArrayPrimitiveType<*>)
                buffer.append(convertArrayName(field.obfName, type.size))
            else
                buffer.append(field.obfName)
            if(field.initialExpression != null){
                buffer.append("=")
                val needCast = type != field.initialExpression!!.type
                if (needCast)
                    buffer.append("(").append(toCType(header, type)).append(")(")
                stringifyExpression(header, buffer, field.initialExpression!!)
                if (needCast)
                    buffer.append(")")
            }
            if(i != fields.lastIndex)
                buffer.append(",")
        }
        buffer.append(";")
    }

    protected open fun stringifyFunctionStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        statement: FunctionStatement
    ){
        val function = statement.function
        if(function.name == "main"){
            stringifyMainFunctionDefinition(header, buffer, function)
            buffer.append("{")

            // Inputs struct
            if(useLocalStruct) {
                buffer.append("__in _v={")
                buffers.forEachIndexed { index, field ->
                    buffer.append("__v").append(field.obfName)
                    if (index != buffers.lastIndex || locals.isNotEmpty())
                        buffer.append(",")
                }
                locals.forEachIndexed { index, field ->
                    stringifyExpression(header, buffer, field.initialExpression!!, false)
                    if (index != buffers.lastIndex)
                        buffer.append(",")
                }
                buffer.append("};")
                buffer.append("${stringifyModifiersInLocalsStruct()} __in*__v=&_v;")
            }
            stringifyMainFunctionBody(header, buffer, function)
            stringifyScopeStatement(header, buffer, statement.function.body!!, false)
            buffer.append("}")
        }else {
            if(statement is FunctionDefinitionStatement && !useFunctionDefs)
                return

            val modifiers = stringifyModifiersInGlobal(function)
            if(modifiers.isNotEmpty())
                buffer.append(modifiers).append(" ")

            val args = function.arguments.map { convertToFuncArg(header, it) }.toMutableList()

            // If function is class member, then add context argument
            if(useStructClasses && contextClass != null)
                args.add(0, "${contextClass!!.obfName} *__s")

            if(useLocalStruct)
                args.add(0, "${stringifyModifiersInLocalsStruct()} __in *__v")

            val type = function.returnType
            val pureName = if (type is ArrayPrimitiveType<*>)
                convertArrayName(function.obfName, type.size) else function.obfName
            val name = if(contextClass != null)
                "${contextClass!!.obfName}_${pureName}" else pureName

            appendCFunctionDefinition(
                buffer = buffer,
                type = toCType(header, type),
                name = name,
                args = args
            )
            if(statement !is FunctionDefinitionStatement)
                stringifyScopeStatement(header, buffer, statement.function.body!!, true)
            else
                buffer.append(";")
        }
    }

    protected open fun stringifyClassStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        classStatement: ClassStatement
    ){
        if(!useStructClasses)
            return
        val clazz = classStatement.classObj

        buffer.append("typedef struct{")
        clazz.variables.values.joinTo(buffer, separator = ";"){
            convertToFuncArg(header, it)
        }
        if(clazz.variables.isNotEmpty())
            buffer.append(";")
        buffer.append("}").append(clazz.obfName).append(";")

        contextClass = clazz
        clazz.body?.scopeObj?.statements?.forEach {
            if(it is FunctionStatement)
                stringifyFunctionStatement(header, buffer, it)
        }
        contextClass = null
    }

    /* ================== *\
           Expressions
    \* ================== */

    protected open fun stringifyArrayDefinitionExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ArrayDefinitionExpression
    ){
        if(useArrayStruct && useArrayStructCast)
            buffer.append("(").append(toCType(header, expression.type)).append(")")

        buffer.append("{")
        expression.elements.forEachIndexed { i, e ->
            stringifyExpression(header, buffer, e)
            if (i < expression.elements.size - 1)
                buffer.append(",")
        }
        buffer.append("}")
    }

    protected open fun stringifyAxBExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: AxBExpression
    ){
        stringifyExpression(header, buffer, expression.left)
        buffer.append(expression.operator.token)
        stringifyExpression(header, buffer, expression.right)
    }

    protected open fun stringifyAxExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: AxExpression
    ){
        stringifyExpression(header, buffer, expression.left)
        buffer.append(expression.operator.token)
    }

    protected open fun stringifyXBExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: XBExpression
    ){
        buffer.append(expression.operator.token)
        stringifyExpression(header, buffer, expression.right)
    }

    protected open fun stringifyArrayAccessExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ArrayAccessExpression
    ){
        stringifyExpression(header, buffer, expression.array)

        // If array is const-sized, and 'useArrayStruct' is enabled, then access struct field
        if(useArrayStruct && expression.array.type.isConstArray)
            buffer.append(".v")

        buffer.append("[")
        stringifyExpression(header, buffer, expression.index)
        buffer.append("]")
    }

    protected open fun stringifyFunctionCallExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: FunctionCallExpression,
    ){
        val function = expression.function
        val isPredefined = function.obfName in predefinedMathFunctions

        if(isPredefined && function.returnType != VOID)
            buffer.append("(").append(toCType(header, function.returnType)).append(")")
        if(expression.obj != null) {
            if(useStructClasses) {
                val className = (expression.obj.type as ClassType).className
                val clazz = expression.function.scope!!.findClass(className)!!
                buffer.append(clazz.obfName).append("_")
            }else {
                stringifyExpression(header, buffer, expression.obj)
                buffer.append(".")
            }
        }
        buffer.append(if(isPredefined) convertPredefinedFunctionName(expression) else function.obfName)
        buffer.append("(")

        val arguments = arrayListOf<String>()

        if(useLocalStruct && !isPredefined)
            arguments += "__v"

        if(useStructClasses && expression.obj != null) {
            val objBuffer = StringBuilder()
            stringifyExpression(header, objBuffer, expression.obj)
            arguments += "&$objBuffer"
        }

        expression.arguments.forEachIndexed { i, arg ->
            val argBuffer = StringBuilder()

            val needCast = arg.type != function.argumentsTypes[i]
            if(isPredefined && needCast)
                argBuffer.append("(").append(toCType(header, function.argumentsTypes[i])).append(")(")
            stringifyExpression(header, argBuffer, arg)
            if(isPredefined && needCast)
                argBuffer.append(")")

            arguments += argBuffer.toString()
        }
        arguments.joinTo(buffer, separator = ",")
        buffer.append(")")
    }

    open fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) =
        functionExpression.function.obfName

    open fun convertPredefinedFieldName(field: GPField) =
        field.obfName

    protected open fun stringifyConstExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ConstExpression
    ){
        buffer.append(expression.lexeme.text)
        if(expression.type.isFloating){
            if(!expression.lexeme.text.contains("."))
                buffer.append(".0")
            buffer.append("f")
        }
    }

    protected open fun stringifyBracketExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: BracketExpression
    ){
        buffer.append("(")
        stringifyExpression(header, buffer, expression.wrapped)
        buffer.append(")")
    }

    protected open fun stringifyCastExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: CastExpression
    ){
        buffer.append("(").append(toCType(header, expression.type)).append(")")
        stringifyExpression(header, buffer, expression.right)
    }

    protected open fun stringifyFieldExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: FieldExpression
    ){
        if(useLocalStruct && (expression.field.isExtern || expression.field.isLocal))
            buffer.append("__v->")
        if(useStructClasses && contextClass != null && contextClass!!.variables[expression.field.name] == expression.field)
            buffer.append("__s->")

        val name = expression.field.obfName
        if(expression.obj != null) {
            stringifyExpression(header, buffer, expression.obj)
            buffer.append(".")
        }
        buffer.append(if(name in allPredefinedFields) convertPredefinedFieldName(expression.field) else name)
    }

    protected open fun stringifyClassCreationExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ClassCreationExpression
    ){
        if(useArrayStruct && useArrayStructCast)
            buffer.append("(").append(toCType(header, expression.type)).append(")")

        buffer.append("{")
        expression.arguments.forEachIndexed { i, e ->
            stringifyExpression(header, buffer, e)
            if (i < expression.arguments.size - 1)
                buffer.append(",")
        }
        buffer.append("}")
    }

    protected open fun convertToFuncArg(header: MutableMap<String, String>, field: GPField): String{
        val buffer = StringBuilder()
        val modifiers = stringifyModifiersInArg(field)
        if(modifiers.isNotEmpty())
            buffer.append(modifiers).append(" ")

        buffer.append(toCType(header, field.type)).append(" ")
        if(field.type is ArrayPrimitiveType<*>)
            buffer.append(convertArrayName(field.obfName, field.type.size))
        else buffer.append(field.obfName)
        return buffer.toString()
    }

    protected open fun toCType(header: MutableMap<String, String>, type: PrimitiveType): String {
        return if(useArrayStruct && type.isConstArray)
            addArrayStruct(header, type as ArrayPrimitiveType<*>)
        else convertType(type)
    }

    protected open fun convertType(type: PrimitiveType) = when(type) {
        is VoidType -> "void"
        is FloatType, is FloatArrayType -> "float"
        is IntType, is IntArrayType -> "int"
        is ByteType, is ByteArrayType, is BooleanType -> "char"
        else -> type.toString()
    }

    protected open fun convertArrayName(name: String, size: Int) =
        if(size == -1) "*$name"
        else if(useArrayStruct) name
        else "$name[$size]"

    protected open fun addArrayStruct(header: MutableMap<String, String>, type: ArrayPrimitiveType<*>): String{
        val name = "${type.single}_${type.size}"
        if(name !in header)
            header[name] = "typedef struct{${convertType(type.single)} v[${type.size}];}$name;"
        return name
    }

    protected open fun appendCFunctionDefinition(
        buffer: StringBuilder,
        type: String,
        name: String,
        args: List<String>
    ){
        buffer.append(type).append(" ").append(name).append("(")
        args.joinTo(buffer, separator = ",")
        buffer.append(")")
    }
}

