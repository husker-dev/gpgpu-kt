package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.Type

class GPGPUASTException(
    text: String,
    val lexeme: Lexeme,
    val codeBlock: String
): Exception(text)

fun compilationError(text: String, lexeme: Lexeme, originalCode: String): Exception {
    val codeLines = originalCode.split("\n")
    fun linePrefix(i: Int?) = if(i == null) " ".repeat(7)
    else "${" ".repeat(6 - i.toString().length)}$i|"

    val lines = arrayListOf<String>()
    if(lexeme.lineIndex > 0)
        lines += "${linePrefix(lexeme.lineIndex)}${codeLines[lexeme.lineIndex-1]}"
    lines += "${linePrefix(lexeme.lineIndex+1)}${codeLines[lexeme.lineIndex]}"
    lines += "${linePrefix(null)}${" ".repeat(lexeme.inlineIndex)}^ here"
    if(lexeme.lineIndex < codeLines.lastIndex)
        lines += "${linePrefix(lexeme.lineIndex+2)}${codeLines[lexeme.lineIndex+1]}"

    return GPGPUASTException("ERROR[${lexeme.lineIndex+1},${lexeme.inlineIndex+1}]: $text\n${lines.joinToString("\n")}",
        lexeme, originalCode)
}

fun compilationError(text: String, lineIndex: Int, inlineIndex: Int, originalCode: String) = compilationError(
    text = text,
    originalCode = originalCode,
    lexeme = Lexeme("", Lexeme.Type.SPECIAL, lineIndex, inlineIndex)
)

fun unexpectedEofException(lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Unexpected end of file",
    originalCode = originalCode,
    lexeme = Lexeme(lexeme.text, lexeme.type, lexeme.lineIndex, lexeme.inlineIndex + lexeme.text.length)
)

fun variableAlreadyDefinedException(name: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Variable '$name' is already defined in the scope",
    originalCode = originalCode,
    lexeme = lexeme
)

fun functionAlreadyDefinedException(name: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Function '$name' is already defined",
    originalCode = originalCode,
    lexeme = lexeme
)

fun expectedTypeException(expected: Type, actual: Type, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Expected '${expected.text}' but found '${actual.text}'",
    originalCode = originalCode,
    lexeme = lexeme
)

fun cantUseOperatorException(operator: Operator, type: Type, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Use of '${operator.token}' with type '${type.text}' is not allowed",
    originalCode = originalCode,
    lexeme = lexeme
)

fun functionIsNotDefined(name: String, argumentTypes: List<Type>, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Function '$name(${if(argumentTypes.isEmpty()) "void" else argumentTypes.joinToString(", ") { it.text }})' is not defined in this scope",
    originalCode = originalCode,
    lexeme = lexeme
)

fun fieldIsNotDefined(name: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Field '$name' is not defined in this scope",
    originalCode = originalCode,
    lexeme = lexeme
)

fun unknownExpression(lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Unknown expression '${lexeme.text}'",
    originalCode = originalCode,
    lexeme = lexeme
)