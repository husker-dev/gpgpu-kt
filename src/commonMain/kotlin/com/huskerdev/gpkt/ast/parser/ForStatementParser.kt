package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseForStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
): ForStatement{
    var i = from + 1
    if(lexemes[i].text != "(")
        throw compilationError("Expected for body", lexemes[i], codeBlock)

    // Find closing bracket
    var r = i
    var brackets = 1
    while(brackets != 0){
        val text = lexemes[++r].text
        if(text == "(" || text == "{" || text == "[") brackets++
        if(text == ")" || text == "}" || text == "]") brackets--
    }

    // Head scope
    val headScope = Scope(scope, Type.VOID)
    i = parseScope(headScope, lexemes, codeBlock, i+1, r)

    /*
    val semicolons = headScope.statements.size
    if(semicolons != 2)
        throw compilationError("Expected three statements", lexemes[i+2], codeBlock)
     */

    // block 1
    val initialization = headScope.statements[0]

    // block 2
    val condition = headScope.statements[1]
    /*
    if(condition !is ExpressionStatement || !condition.expression.type.isLogical)
        throw compilationError("Expected boolean in condition", lexemes[i+1], codeBlock)
     */

    // block 3
    val iteration = headScope.statements.getOrElse(2) {
        EmptyStatement(condition.lexemeIndex + condition.lexemeLength, 0)
    }

    // body
    val body = Scope(scope, Type.VOID).apply {
        if(initialization is FieldStatement)
            initialization.fields.forEach { addField(it, lexemes[i], codeBlock) }
        i = if(lexemes[i].text == "{")
            parseScope(this, lexemes, codeBlock, i+1, lexemes.size)
        else parseScope(this, lexemes, codeBlock, i, findExpressionEnd(lexemes, i)) + 1
    }

    return ForStatement(initialization, condition, iteration, body, from, i - from)
}