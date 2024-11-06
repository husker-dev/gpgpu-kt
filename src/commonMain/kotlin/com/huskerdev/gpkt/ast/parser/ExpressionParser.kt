package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.operatorTokens
import com.huskerdev.gpkt.ast.lexer.primitives
import com.huskerdev.gpkt.ast.objects.GPClass
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.*

fun parseExpression(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int = findExpressionEnd(from, lexemes, codeBlock)
): Expression? {
    //println("parse: " + lexemes.subList(from, to).joinToString(" ") { it.text })

    if(to - from == 0)
        return null

    if(lexemes[from].text == "("){
        var brackets = 1
        var r = from
        while(brackets != 0 && r < to){
            val text = lexemes[++r].text
            if(text == "(") brackets++
            if(text == ")") brackets--
        }
        if(r == to-1)
            return parseExpression(scope, lexemes, codeBlock, from+1, to-1)?.run {
                BracketExpression(this, from, to - from)
            }
    }

    if(lexemes[from].text == "{" && lexemes[to-1].text == "}"){
        val elements = arrayListOf<Expression>()
        var r = from + 1
        while(r < to && to - r > 1){
            val element = parseExpression(scope, lexemes, codeBlock, r)!!
            elements += element
            r += element.lexemeLength + 1
        }
        return ArrayDefinitionExpression(elements.toTypedArray(), from, to - from)
    }

    Operator.sortedReverse.forEach { operator ->
        when(operator.usage) {
            Operator.Usage.AxB -> {
                val token = operator.token
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if (i != from && i != to-1 && lexeme.text == token && lexemes[i-1].text !in operatorTokens) {
                        val left = parseExpression(scope, lexemes, codeBlock, from, i)!!
                        val right = parseExpression(scope, lexemes, codeBlock, i + 1, to)!!

                        operator.checkOpAxB(left, right, lexemes[i], lexemes[i+1], codeBlock)
                        val type = operator.operateTypeAxB(left.type, right.type)

                        return unpackedAxB(scope, operator, left, right, type, from, to - from)
                    }
                }
            }
            Operator.Usage.Ax -> {
                val token = operator.token
                if (lexemes[to-1].text == token) {
                    val left = parseExpression(scope, lexemes, codeBlock, from, to-1)!!

                    operator.checkOpAx(left, lexemes[to-1], codeBlock)
                    val type = operator.operateTypeAx(left.type)

                    return unpackedAx(scope, operator, left, type, from, to - from)
                }
            }
            Operator.Usage.xB -> {
                val token = operator.token
                if (lexemes[from].text == token) {
                    val right = parseExpression(scope, lexemes, codeBlock, from + 1, to)!!

                    operator.checkOpXB(right, lexemes[from], codeBlock)
                    val type = operator.operateTypeXB(right.type)

                    return unpackedXB(scope, operator, right, type, from, to - from)
                }
            }
            Operator.Usage.ARRAY_ACCESS -> {
                if(lexemes[to-1].text == "]"){
                    val leftBracket = findExpressionStart(to-2, lexemes, codeBlock)
                    if(lexemes[leftBracket].text != "[")
                        throw expectedException("[", lexemes[leftBracket], codeBlock)

                    val array = parseExpression(scope, lexemes, codeBlock, from, leftBracket)!!
                    if(array.type !is ArrayPrimitiveType<*>)
                        throw expectedException("<array>", array.type.toString(), lexemes[from], codeBlock)

                    val indexExpression = parseExpression(scope, lexemes, codeBlock, leftBracket+1)
                        ?: throw expectedException("index", lexemes[leftBracket+1], codeBlock)
                    if(!indexExpression.type.isInteger)
                        throw expectedTypeException(INT, indexExpression.type, lexemes[leftBracket+1], codeBlock)

                    return unpackedArrayAccess(scope, array, indexExpression, from, to - from)
                }
            }
            Operator.Usage.CAST -> {
                if (lexemes[from].text == "(" &&
                    lexemes[from + 1].text in primitives &&
                    lexemes[from + 2].text == ")"
                ) {
                    val right = parseExpression(scope, lexemes, codeBlock, from+3, to)!!
                    val type = primitivesMap[lexemes[from+1].text] ?:
                        throw cannotCastException(right.type, lexemes[from+1].text, lexemes[from+1], codeBlock)

                    if(type !in PrimitiveType.allowedCastMap || right.type !in PrimitiveType.allowedCastMap[type]!!)
                        throw cannotCastException(right.type, type, lexemes[from+3], codeBlock)
                    return unpackedCast(scope, type, right, from, to - from)
                }
            }
            Operator.Usage.FUNCTION -> {
                if(lexemes[to-1].text != ")")
                    return@forEach

                // Simple case: global function
                if(lexemes[from].type == Lexeme.Type.NAME && lexemes[from+1].text == "("){
                    val lexeme = lexemes[from]
                    val (arguments, types, _) =
                        readArguments(scope, lexemes, codeBlock, from + 2, to)

                    val function = scope.findDefinedFunction(lexeme.text)
                        ?: throw functionIsNotDefined(lexeme.text, types, lexeme, codeBlock)

                    if(!function.canAcceptArguments(types))
                        throw wrongFunctionParameters(function, types, lexeme, codeBlock)

                    return unpackedFunctionCall(scope, null, function, arguments, from, to - from)
                }

                // Access class function
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if (i != from && i != to-1 && lexeme.text == ".") {
                        val left = parseExpression(scope, lexemes, codeBlock, from, i)!!
                        val functionLexeme = lexemes[i + 1]

                        if(left.type !is ClassType)
                            throw compilationError("Member is not an object", lexemes[from], codeBlock)
                        if(functionLexeme.type != Lexeme.Type.NAME)
                            throw expectedException("function name", functionLexeme.text, lexemes[from], codeBlock)

                        val clazz = scope.findDefinedClass((left.type as ClassType).className)!!

                        if(clazz.body == null || functionLexeme.text !in clazz.body.scope.functions)
                            throw compilationError("Function '${functionLexeme.text}' is not defined in class '${clazz.name}'", lexemes[from], codeBlock)
                        val function = clazz.body.scope.functions[functionLexeme.text]!!

                        val (arguments, types, _) =
                            readArguments(scope, lexemes, codeBlock, i + 3, to)

                        if(!function.canAcceptArguments(types))
                            throw wrongFunctionParameters(function, types, lexeme, codeBlock)

                        return unpackedFunctionCall(scope, left, function, arguments, from, to - from)
                    }
                }
            }
            Operator.Usage.FIELD -> {
                if(lexemes[to-1].text == ")")
                    return@forEach

                // Simple case: just field
                if(to - from == 1){
                    val lexeme = lexemes[from]
                    if(lexeme.type == Lexeme.Type.NAME){
                        val field = scope.findDefinedField(lexeme.text)
                            ?: throw fieldIsNotDefined(lexeme.text, lexeme, codeBlock)
                        return FieldExpression(null, field, from, 1)
                    }
                    return createConstExpression(from, lexeme, codeBlock)
                }

                // Access class field
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if (i != from && i != to-1 && lexeme.text == ".") {
                        val left = parseExpression(scope, lexemes, codeBlock, from, i)!!
                        val right = lexemes[i + 1]

                        if(left.type !is ClassType)
                            throw compilationError("Member is not an object", lexemes[from], codeBlock)
                        if(right.type != Lexeme.Type.NAME)
                            throw expectedException("field name", right.text, lexemes[from], codeBlock)

                        val clazz = scope.findDefinedClass((left.type as ClassType).className)!!

                        if(right.text !in clazz.variables)
                            throw compilationError("Field '${right.text}' is not defined in class '${clazz.name}'", lexemes[from], codeBlock)

                        return FieldExpression(left, clazz.variables[right.text]!!, from, to - from)
                    }
                }
            }
            Operator.Usage.NEW -> {
                if(lexemes[from].text == "new"){
                    val classL = lexemes[from + 1]
                    if(classL.type != Lexeme.Type.NAME)
                        throw expectedException("class name", lexemes[from + 1], codeBlock)

                    val classObj = scope.findDefinedClass(classL.text)
                        ?: throw compilationError("Can not find class '${classL.text}'", lexemes[from + 1], codeBlock)

                    if(lexemes[from + 2].text != "(")
                        throw expectedException("arguments block", lexemes[from + 2], codeBlock)

                    val (arguments, types, argsTo) =
                        readArguments(scope, lexemes, codeBlock, from + 3, to)

                    classObj.variablesTypes.forEachIndexed { index, expected ->
                        val actual = types.getOrNull(index)
                            ?: throw expectedException(expected.toString(), "none", lexemes[to - 1], codeBlock)
                        if(expected != actual && !PrimitiveType.canAssignNumbers(expected, actual))
                            throw expectedException(expected.toString(), actual.toString(), lexemes[arguments[index].lexemeIndex], codeBlock)
                    }
                    if(classObj.variablesTypes.size != types.size)
                        throw compilationError("Too many arguments", lexemes[arguments[classObj.variablesTypes.size].lexemeIndex], codeBlock)

                    return ClassCreationExpression(classObj, arguments, from, argsTo - from + 1)
                }
            }
            else -> {}
            // Operator.Usage.CONDITION -> TODO()
        }
    }
    throw unknownExpression(lexemes[from], codeBlock)
}

fun getPrimitiveClassSetter(clazz: GPClass) = when{
    "Float" in clazz.implements -> clazz.body!!.scope.functions["setFloat"]!!
    "Int" in clazz.implements -> clazz.body!!.scope.functions["setInt"]!!
    "Byte" in clazz.implements -> clazz.body!!.scope.functions["setByte"]!!
    "Boolean" in clazz.implements -> clazz.body!!.scope.functions["setBoolean"]!!
    else -> throw UnsupportedOperationException()
}

fun getPrimitiveClassGetter(clazz: GPClass) = when{
    "Float" in clazz.implements -> clazz.body!!.scope.functions["getFloat"]!!
    "Int" in clazz.implements -> clazz.body!!.scope.functions["getInt"]!!
    "Byte" in clazz.implements -> clazz.body!!.scope.functions["getByte"]!!
    "Boolean" in clazz.implements -> clazz.body!!.scope.functions["getBoolean"]!!
    else -> throw UnsupportedOperationException()
}

/**
 *  Unpacks left and right expression if they are classes that implements Float/Int/Byte/Boolean
 */
private fun unpackedAxB(
    scope: GPScope,
    operator: Operator,
    left: Expression,
    right: Expression,
    type: PrimitiveType,
    from: Int,
    length: Int
): Expression {
    val unpackedRight = if(left.type != right.type && right.type is ClassType){
        val clazz = scope.findDefinedClass((right.type as ClassType).className)!!
        FunctionCallExpression(right, getPrimitiveClassGetter(clazz), emptyList())
    } else right

    val unpackedLeft = if(left.type != right.type && left.type is ClassType) {
        val clazz = scope.findDefinedClass((left.type as ClassType).className)!!
        when (operator) {
            // Assignment
            Operator.ASSIGN ->
                return FunctionCallExpression(left, getPrimitiveClassSetter(clazz), arrayListOf(unpackedRight), from, length)

            // Operator and assignment
            Operator.MOD_ASSIGN, Operator.BITWISE_AND_ASSIGN, Operator.BITWISE_OR_ASSIGN, Operator.BITWISE_XOR_ASSIGN,
            Operator.BITWISE_SHIFT_RIGHT_ASSIGN, Operator.BITWISE_SHIFT_LEFT_ASSIGN, Operator.DIVIDE_ASSIGN,
            Operator.MULTIPLY_ASSIGN, Operator.MINUS_ASSIGN, Operator.PLUS_ASSIGN -> {
                return FunctionCallExpression(left, getPrimitiveClassSetter(clazz), arrayListOf(
                    AxBExpression(operator.assignOpToSimple(), type,
                        FunctionCallExpression(left, getPrimitiveClassGetter(clazz), emptyList()),
                        unpackedRight
                    )
                ), from, length)
            }

            // Without assignment
            else -> FunctionCallExpression(left, getPrimitiveClassGetter(clazz), emptyList())
        }
    } else left
    return AxBExpression(operator, type, unpackedLeft, unpackedRight, from, length)
}

/**
 *  Unpacks left expression if they are classes that implements Float/Int/Byte/Boolean
 */
private fun unpackedAx(
    scope: GPScope,
    operator: Operator,
    left: Expression,
    type: PrimitiveType,
    from: Int,
    length: Int
): Expression {
    val unpackedLeft = if(left.type is ClassType) {
        val clazz = scope.findDefinedClass((left.type as ClassType).className)!!
        when (operator) {
            // Operator and assignment
            Operator.INCREASE, Operator.DECREASE -> {
                return FunctionCallExpression(left, getPrimitiveClassSetter(clazz), arrayListOf(
                    AxBExpression(operator.assignOpToSimple(), type,
                        FunctionCallExpression(left, getPrimitiveClassGetter(clazz), emptyList()),
                        ConstExpression(Lexeme("1", Lexeme.Type.INT), INT)
                    )
                ), from, length)
            }
            // Without assignment
            else -> FunctionCallExpression(left, getPrimitiveClassGetter(clazz), emptyList())
        }
    } else left
    return AxExpression(operator, type, unpackedLeft, from, length)
}

/**
 *  Unpacks right expression if they are classes that implements Float/Int/Byte/Boolean
 */
private fun unpackedXB(
    scope: GPScope,
    operator: Operator,
    right: Expression,
    type: PrimitiveType,
    from: Int,
    length: Int
): Expression {
    val unpackedRight = if(right.type is ClassType) {
        val clazz = scope.findDefinedClass((right.type as ClassType).className)!!
        FunctionCallExpression(right, getPrimitiveClassGetter(clazz), emptyList())
    } else right
    return XBExpression(operator, type, unpackedRight, from, length)
}

/**
 *  Unpacks index expression if they are classes that implements Float/Int/Byte/Boolean
 */
private fun unpackedArrayAccess(
    scope: GPScope,
    array: Expression,
    index: Expression,
    from: Int,
    length: Int
): Expression {
    val unpackedIndex = if(index.type is ClassType) {
        val clazz = scope.findDefinedClass((index.type as ClassType).className)!!
        FunctionCallExpression(index, getPrimitiveClassGetter(clazz), emptyList())
    } else index
    return ArrayAccessExpression(array, unpackedIndex, from, length)
}

/**
 *  Unpacks expression if they are classes that implements Float/Int/Byte/Boolean
 */
private fun unpackedCast(
    scope: GPScope,
    type: PrimitiveType,
    expr: Expression,
    from: Int,
    length: Int
): Expression {
    val unpackedExpr = if(expr.type is ClassType) {
        val clazz = scope.findDefinedClass((expr.type as ClassType).className)!!
        FunctionCallExpression(expr, getPrimitiveClassGetter(clazz), emptyList())
    } else expr
    return CastExpression(type, unpackedExpr, from, length)
}

/**
 *  Unpacks argument expressions if they are classes that implements Float/Int/Byte/Boolean
 */
private fun unpackedFunctionCall(
    scope: GPScope,
    obj: Expression?,
    function: GPFunction,
    arguments: List<Expression>,
    from: Int,
    length: Int
): Expression {
    val unpackedArguments = arguments.mapIndexed { i, expr ->
        if(expr.type is ClassType && function.argumentsTypes[i] != expr.type){
            val clazz = scope.findDefinedClass((expr.type as ClassType).className)!!
            FunctionCallExpression(expr, getPrimitiveClassGetter(clazz), emptyList())
        }else expr
    }
    return FunctionCallExpression(obj, function, unpackedArguments, from, length)
}

fun readArguments(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
): Triple<List<Expression>, List<PrimitiveType>, Int>{
    val arguments = mutableListOf<Expression>()
    val argumentTypes = mutableListOf<PrimitiveType>()

    var i = from
    if(lexemes[i].text != ")") {
        while (i < to) {
            val argument = parseExpression(scope, lexemes, codeBlock, i)!!
            arguments += argument
            argumentTypes += argument.type

            i += argument.lexemeLength
            val next = lexemes[i]
            if (next.text == ",") {
                i++
                continue
            } else if (next.text == ")")
                break
            else throw unexpectedSymbolException(next.text, next, codeBlock)
        }
    }
    return Triple(arguments, argumentTypes, i)
}

fun findExpressionEnd(from: Int, lexemes: List<Lexeme>, codeBlock: String): Int{
    var brackets = 0
    var endIndex = from
    while(endIndex < lexemes.size) {
        val lexeme = lexemes[endIndex++]
        val text = lexeme.text
        if (brackets == 0 && (text == "," || text == ";" || text == ")" || text == "]" || text == "}"))
            return endIndex - 1
        if (text == "[" || text == "(" || text == "{") brackets++
        if (text == "]" || text == ")" || text == "}") brackets--
    }
    throw expectedException(";", lexemes.last(), codeBlock)
}

fun findExpressionStart(from: Int, lexemes: List<Lexeme>, codeBlock: String): Int{
    var brackets = 0
    var startIndex = from
    while(startIndex < lexemes.size) {
        val lexeme = lexemes[startIndex--]
        val text = lexeme.text
        if (brackets == 0 && (text == "," || text == ";" || text == "(" || text == "["))
            return startIndex + 1
        if (text == "[" || text == "(" || text == "{") brackets++
        if (text == "]" || text == ")" || text == "}") brackets--
    }
    throw unknownExpression(lexemes[startIndex], codeBlock)
}

fun createConstExpression(index: Int, lexeme: Lexeme, codeBlock: String) = when (lexeme.type) {
    Lexeme.Type.FLOAT   -> ConstExpression(lexeme, FLOAT, index, 1)
    Lexeme.Type.INT     -> ConstExpression(lexeme, INT, index, 1)
    Lexeme.Type.BYTE    -> ConstExpression(lexeme, BYTE, index, 1)
    Lexeme.Type.LOGICAL -> ConstExpression(lexeme, BOOLEAN, index, 1)
    else -> throw unknownExpression(lexeme, codeBlock)
}

fun nextLexemeIgnoringBrackets(
    index: Int,
    lexemes: List<Lexeme>,
): Int {
    var i = index
    var brackets = 0
    while(i < lexemes.size) {
        val l = lexemes[i++]
        if (l.text == "[" || l.text == "(" || l.text == "{") brackets++
        if (l.text == "]" || l.text == ")" || l.text == "}") brackets--
        if (brackets == 0)
            return i
    }
    return i
}

inline fun foreachLexemeIgnoringBrackets(from: Int, to: Int, lexemes: List<Lexeme>, block: (Int, Lexeme) -> Unit){
    var i = from
    while(i < to) {
        block(i, lexemes[i])
        i = nextLexemeIgnoringBrackets(i, lexemes)
    }
}