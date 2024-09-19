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
        when(operator.usage){
            Operator.Usage.AxB -> {
                val token = operator.token
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if(lexeme.text == token){
                        val left = parseExpression(scope, lexemes, codeBlock, from, i)!!
                        val right = parseExpression(scope, lexemes, codeBlock, i+1, to)!!

                        if(operator.flags and FLAG_LOGICAL_TYPES == FLAG_LOGICAL_TYPES){
                            if(!left.type.isLogical) throw expectedTypeException(Type.BOOLEAN, left.type, lexeme, codeBlock)
                            if(!right.type.isLogical) throw expectedTypeException(Type.BOOLEAN, right.type, lexemes[i+1], codeBlock)
                        }else if(operator.flags and FLAG_NUMERIC_TYPES == FLAG_NUMERIC_TYPES){
                            if(!left.type.isNumber) throw cantUseOperatorException(operator, left.type, lexeme, codeBlock)
                            if(!right.type.isNumber) throw cantUseOperatorException(operator, right.type, lexemes[i+1], codeBlock)
                            if(left.type != right.type) throw expectedTypeException(left.type, right.type, lexemes[i+1], codeBlock)
                        }else if(operator.flags and FLAG_EQUAL_TYPES == FLAG_EQUAL_TYPES)
                            if(left.type != right.type) throw expectedTypeException(left.type, right.type, lexemes[i+1], codeBlock)

                        val type = if(operator.flags and FLAG_RETURNS_BOOLEAN == FLAG_RETURNS_BOOLEAN)
                            Type.BOOLEAN
                        else left.type

                        return AxBExpression(operator, type, left, right, from, to - from)
                    }
                }
            }
            Operator.Usage.Ax -> {
                val token = operator.token
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if(lexeme.text == token){
                        val left = parseExpression(scope, lexemes, codeBlock, i-1, i)!!

                        if(operator.flags and FLAG_FIELD_TYPE == FLAG_FIELD_TYPE)
                            if(left !is FieldExpression) throw compilationError("Expected variable", lexemes[i-1], codeBlock)
                        return AxExpression(operator, left.type, left, from, 2)
                    }
                }
            }
            Operator.Usage.FUNCTION -> {
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if(lexeme.type == Lexeme.Type.NAME && lexemes[i+1].text == "("){
                        val arguments = mutableListOf<Expression>()
                        val argumentTypes = mutableListOf<Type>()

                        var r = i+2
                        while(r < to){
                            val argument = parseExpression(scope, lexemes, codeBlock, r)!!
                            arguments += argument
                            argumentTypes += argument.type

                            r += argument.lexemeLength
                            val next = lexemes[r]
                            if(next.text == ",") {
                                r++
                                continue
                            } else if(next.text == ")") {
                                r++
                                break
                            } else throw compilationError("Unexpected symbol '${next.text}'", next, codeBlock)
                        }
                        val function = scope.findDefinedFunction(lexeme.text, argumentTypes) ?:
                            throw functionIsNotDefined(lexeme.text, argumentTypes, lexeme, codeBlock)

                        return FunctionCallExpression(operator, function, arguments, i, i - r)
                    }
                }
            }
            Operator.Usage.ARRAY_ACCESS -> {
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if (lexeme.text == "[") {
                        val array = parseExpression(scope, lexemes, codeBlock, i - 1, i)!!
                        if(!array.type.isArray || array !is FieldExpression)
                            throw compilationError("cannot use '[]' with type '${array.type.text}'", lexemes[i-1], codeBlock)

                        val indexExpression = parseExpression(scope, lexemes, codeBlock, i+1) ?:
                        throw compilationError("Expected index", lexemes[i+1], codeBlock)
                        if(indexExpression.type != Type.INT)
                            throw expectedTypeException(Type.INT, indexExpression.type, lexemes[i+1], codeBlock)

                        return ArrayAccessExpression(array.field, indexExpression, i-1, 3 + indexExpression.lexemeLength)
                    }
                }
            }
            Operator.Usage.CAST -> {
                foreachLexemeIgnoringBrackets(from, to, lexemes) { i, lexeme ->
                    if (i < lexemes.lastIndex - 2 && lexeme.text == "(" && lexemes[i+1].text in primitives && lexemes[i+2].text == ")") {
                        val type = Type.map[lexemes[i+1].text] ?:
                            throw compilationError("Cannot cast to unknown type '${lexemes[i+1].text}'", lexemes[i+1], codeBlock)

                        val right = parseExpression(scope, lexemes, codeBlock, i+3)!!
                        if(type !in Type.castMap || right.type !in Type.castMap[type]!!)
                            throw compilationError("Cannot cast '${right.type.text}' to '${type.text}'", lexemes[i+1], codeBlock)

                        return CastExpression(type, right, i, 2 + right.lexemeLength)
                    }
                }
            }
            else -> {}
            /*
            Operator.Usage.xB -> TODO()
            Operator.Usage.CONDITION -> TODO()
             */
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