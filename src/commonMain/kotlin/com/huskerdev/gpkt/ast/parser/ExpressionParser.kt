package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.primitives
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.ast.types.*

fun parseExpression(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int = findExpressionEnd(lexemes, from)
): Expression? {
    //println("parse: " + lexemes.subList(from, to).joinToString(" ") { it.text })
    if(lexemes[from].text == "(" && lexemes[to-1].text == ")")
        return parseExpression(scope, lexemes, codeBlock, from+1, to-1)?.run {
            BracketExpression(this, from, to-1)
        }

    if(to - from == 0)
        return null

    if(to - from == 1){
        val lexeme = lexemes[from]
        if(lexeme.type == Lexeme.Type.NAME){
            val field = scope.findDefinedField(lexeme.text) ?:
            throw fieldIsNotDefined(lexeme.text, lexeme, codeBlock)
            return FieldExpression(field, from, 1)
        }
        return createConstExpression(from, lexeme, codeBlock)
    }

    Operator.sortedReverse.forEach { operator ->
        when(operator.usage) {
            Operator.Usage.AxB -> {
                val token = operator.token
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if (i != from && lexeme.text == token) {
                        val left = parseExpression(scope, lexemes, codeBlock, from, i)!!
                        val right = parseExpression(scope, lexemes, codeBlock, i + 1, to)!!

                        if (operator.flags and FLAG_LOGICAL_TYPES == FLAG_LOGICAL_TYPES) {
                            if (!left.type.isLogical) throw expectedTypeException(Type.BOOLEAN, left.type, lexeme, codeBlock)
                            if (!right.type.isLogical) throw expectedTypeException(Type.BOOLEAN, right.type, lexemes[i + 1], codeBlock)
                        } else if (operator.flags and FLAG_NUMERIC_TYPES == FLAG_NUMERIC_TYPES) {
                            if (!left.type.isNumber) throw cantUseOperatorException(operator, left.type, lexeme, codeBlock)
                            if (!right.type.isNumber) throw cantUseOperatorException(operator, right.type, lexemes[i + 1], codeBlock)
                            if (left.type != right.type) throw expectedTypeException(left.type, right.type, lexemes[i + 1], codeBlock)
                        } else if (operator.flags and FLAG_INT_TYPE == FLAG_INT_TYPE) {
                            if (left.type != Type.INT) throw cantUseOperatorException(operator, left.type, lexeme, codeBlock)
                            if (right.type != Type.INT) throw cantUseOperatorException(operator, right.type, lexemes[i + 1], codeBlock)
                        } else if (operator.flags and FLAG_EQUAL_TYPES == FLAG_EQUAL_TYPES)
                            if (left.type != right.type) throw expectedTypeException(left.type, right.type, lexemes[i + 1], codeBlock)

                        val type = if(operator.flags and FLAG_RETURNS_BOOLEAN == FLAG_RETURNS_BOOLEAN)
                            Type.BOOLEAN
                        else left.type

                        return AxBExpression(operator, type, left, right, from, to - from)
                    }
                }
            }
            Operator.Usage.Ax -> {
                val token = operator.token

                if (lexemes[to-1].text == token) {
                    val left = parseExpression(scope, lexemes, codeBlock, from, to-1)!!

                    if (operator.flags and FLAG_FIELD_TYPE == FLAG_FIELD_TYPE)
                        if (left !is FieldExpression) throw compilationError("Expected variable", lexemes[from], codeBlock)
                    return AxExpression(operator, left.type, left, from, to - from)
                }
            }
            Operator.Usage.xB -> {
                val token = operator.token

                if (lexemes[from].text == token) {
                    val right = parseExpression(scope, lexemes, codeBlock, from + 1, to)!!
                    if (operator.flags and FLAG_LOGICAL_TYPES == FLAG_LOGICAL_TYPES)
                        if (!right.type.isLogical) throw cantUseOperatorException(operator, right.type, lexemes[from + 1], codeBlock)
                    if (operator.flags and FLAG_INT_TYPE == FLAG_INT_TYPE)
                        if (right.type != Type.INT) throw cantUseOperatorException(operator, right.type, lexemes[from + 1], codeBlock)

                    return XBExpression(operator, right.type, right, from, to - from)
                }
            }
            Operator.Usage.FUNCTION -> {
                val lexeme = lexemes[from]
                if(lexeme.type == Lexeme.Type.NAME && lexemes[from+1].text == "("){
                    val arguments = mutableListOf<Expression>()
                    val argumentTypes = mutableListOf<Type>()

                    var r = from+2
                    while(r < to){
                        val argument = parseExpression(scope, lexemes, codeBlock, r)!!
                        arguments += argument
                        argumentTypes += argument.type

                        r += argument.lexemeLength
                        val next = lexemes[r]
                        if(next.text == ",") {
                            r++
                            continue
                        } else if(next.text == ")")
                            break
                        else throw compilationError("Unexpected symbol '${next.text}'", next, codeBlock)
                    }
                    val function = scope.findDefinedFunction(lexeme.text, argumentTypes)
                        ?: throw functionIsNotDefined(lexeme.text, argumentTypes, lexeme, codeBlock)

                    return FunctionCallExpression(operator, function, arguments, from, to - from)
                }
            }
            Operator.Usage.ARRAY_ACCESS -> {
                if(lexemes[to-1].text == "]"){
                    val leftBracket = findExpressionStart(lexemes, to-2)
                    if(lexemes[leftBracket].text != "[")
                        throw compilationError("Expected [", lexemes[leftBracket], codeBlock)

                    val array = parseExpression(scope, lexemes, codeBlock, from, leftBracket)!!
                    if(!array.type.isArray || array !is FieldExpression)
                        throw compilationError("Expected <array> but found '${array.type.text}'", lexemes[from], codeBlock)

                    val indexExpression = parseExpression(scope, lexemes, codeBlock, leftBracket+1)
                        ?: throw compilationError("Expected index", lexemes[leftBracket+1], codeBlock)
                    if(indexExpression.type != Type.INT)
                        throw expectedTypeException(Type.INT, indexExpression.type, lexemes[leftBracket+1], codeBlock)

                    return ArrayAccessExpression(array.field, indexExpression, from, to - from)
                }
            }
            Operator.Usage.CAST -> {
                if (lexemes[from].text == "(" &&
                    lexemes[from + 1].text in primitives &&
                    lexemes[from + 2].text == ")"
                ) {
                    val type = Type.map[lexemes[from+1].text] ?:
                        throw compilationError("Cannot cast to unknown type '${lexemes[from+1].text}'", lexemes[from+1], codeBlock)

                    val right = parseExpression(scope, lexemes, codeBlock, from+3, to)!!
                    if(type !in Type.castMap || right.type !in Type.castMap[type]!!)
                        throw compilationError("Cannot cast '${right.type.text}' to '${type.text}'", lexemes[from+3], codeBlock)
                    return CastExpression(type, right, from, to - from)
                }
            }
            else -> {}
            // Operator.Usage.CONDITION -> TODO()
        }
    }

    throw unknownExpression(lexemes[from], codeBlock)
}


fun findExpressionEnd(lexemes: List<Lexeme>, from: Int): Int{
    var brackets = 0
    var endIndex = from
    while(endIndex < lexemes.size) {
        val lexeme = lexemes[endIndex++]
        val text = lexeme.text
        if (brackets == 0 && (text == "," || text == ";" || text == ")" || text == "]")) {
            endIndex--
            break
        }
        if (text == "[" || text == "(" || text == "{") brackets++
        if (text == "]" || text == ")" || text == "}") brackets--
    }
    return endIndex
}

fun findExpressionStart(lexemes: List<Lexeme>, from: Int): Int{
    var brackets = 0
    var startIndex = from
    while(startIndex < lexemes.size) {
        val lexeme = lexemes[startIndex--]
        val text = lexeme.text
        if (brackets == 0 && (text == "," || text == ";" || text == "(" || text == "[")) {
            startIndex++
            break
        }
        if (text == "[" || text == "(" || text == "{") brackets++
        if (text == "]" || text == ")" || text == "}") brackets--
    }
    return startIndex
}

fun createConstExpression(index: Int, lexeme: Lexeme, codeBlock: String) = when (lexeme.type) {
    Lexeme.Type.NUMBER                -> ConstExpression(lexeme, Type.INT, index, 1)
    Lexeme.Type.NUMBER_FLOATING_POINT -> ConstExpression(lexeme, Type.FLOAT, index, 1)
    Lexeme.Type.LOGICAL               -> ConstExpression(lexeme, Type.BOOLEAN, index, 1)
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