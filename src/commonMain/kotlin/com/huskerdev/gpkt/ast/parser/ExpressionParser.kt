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
    to: Int = findExpressionEnd(from, lexemes, codeBlock)
): Expression? {
    //println("parse: " + lexemes.subList(from, to).joinToString(" ") { it.text })

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
                    if (i != from && i != to-1 && lexeme.text == token) {
                        val left = parseExpression(scope, lexemes, codeBlock, from, i)!!
                        val right = parseExpression(scope, lexemes, codeBlock, i + 1, to)!!

                        operator.checkOpAxB(left, right, lexemes[i], lexemes[i+1], codeBlock)
                        val type = operator.operateTypeAxB(left.type, right.type)

                        return AxBExpression(operator, type, left, right, from, to - from)
                    }
                }
            }
            Operator.Usage.Ax -> {
                val token = operator.token

                if (lexemes[to-1].text == token) {
                    val left = parseExpression(scope, lexemes, codeBlock, from, to-1)!!

                    operator.checkOpAx(left, lexemes[to-1], codeBlock)
                    val type = operator.operateTypeAx(left.type)

                    return AxExpression(operator, type, left, from, to - from)
                }
            }
            Operator.Usage.xB -> {
                val token = operator.token

                if (lexemes[from].text == token) {
                    val right = parseExpression(scope, lexemes, codeBlock, from + 1, to)!!

                    operator.checkOpXB(right, lexemes[from], codeBlock)
                    val type = operator.operateTypeXB(right.type)

                    return XBExpression(operator, type, right, from, to - from)
                }
            }
            Operator.Usage.FUNCTION -> {
                val lexeme = lexemes[from]
                if(lexeme.type == Lexeme.Type.NAME && lexemes[from+1].text == "("){
                    val arguments = mutableListOf<Expression>()
                    val argumentTypes = mutableListOf<Type>()

                    if(lexemes[from+2].text != ")") {
                        var r = from + 2
                        while (r < to) {
                            val argument = parseExpression(scope, lexemes, codeBlock, r)!!
                            arguments += argument
                            argumentTypes += argument.type

                            r += argument.lexemeLength
                            val next = lexemes[r]
                            if (next.text == ",") {
                                r++
                                continue
                            } else if (next.text == ")")
                                break
                            else throw unexpectedSymbolException(next.text, next, codeBlock)
                        }
                    }
                    val function = scope.findDefinedFunction(lexeme.text, argumentTypes)
                        ?: throw functionIsNotDefined(lexeme.text, argumentTypes, lexeme, codeBlock)

                    return FunctionCallExpression(operator, function, arguments, from, to - from)
                }
            }
            Operator.Usage.ARRAY_ACCESS -> {
                if(lexemes[to-1].text == "]"){
                    val leftBracket = findExpressionStart(to-2, lexemes, codeBlock)
                    if(lexemes[leftBracket].text != "[")
                        throw expectedException("[", lexemes[leftBracket], codeBlock)

                    val array = parseExpression(scope, lexemes, codeBlock, from, leftBracket)!!
                    if(!array.type.isArray || array !is FieldExpression)
                        throw expectedException("<array>", array.type.text, lexemes[from], codeBlock)

                    val indexExpression = parseExpression(scope, lexemes, codeBlock, leftBracket+1)
                        ?: throw expectedException("index", lexemes[leftBracket+1], codeBlock)
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
                    val right = parseExpression(scope, lexemes, codeBlock, from+3, to)!!
                    val type = Type.map[lexemes[from+1].text] ?:
                        throw cannotCastException(right.type, lexemes[from+1].text, lexemes[from+1], codeBlock)

                    if(type !in Type.allowedCastMap || right.type !in Type.allowedCastMap[type]!!)
                        throw cannotCastException(right.type, type, lexemes[from+3], codeBlock)
                    return CastExpression(type, right, from, to - from)
                }
            }
            else -> {}
            // Operator.Usage.CONDITION -> TODO()
        }
    }

    throw unknownExpression(lexemes[from], codeBlock)
}


fun findExpressionEnd(from: Int, lexemes: List<Lexeme>, codeBlock: String): Int{
    var brackets = 0
    var endIndex = from
    while(endIndex < lexemes.size) {
        val lexeme = lexemes[endIndex++]
        val text = lexeme.text
        if (brackets == 0 && (text == "," || text == ";" || text == ")" || text == "]"))
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
    Lexeme.Type.DOUBLE  -> ConstExpression(lexeme, Type.DOUBLE, index, 1)
    Lexeme.Type.FLOAT   -> ConstExpression(lexeme, Type.FLOAT, index, 1)
    Lexeme.Type.INT     -> ConstExpression(lexeme, Type.INT, index, 1)
    Lexeme.Type.BYTE    -> ConstExpression(lexeme, Type.BYTE, index, 1)
    Lexeme.Type.LOGICAL -> ConstExpression(lexeme, Type.BOOLEAN, index, 1)
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