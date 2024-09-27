package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.Type

class GPCompilationException(
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

    return GPCompilationException("ERROR[${lexeme.lineIndex+1},${lexeme.inlineIndex+1}]: $text\n${lines.joinToString("\n")}",
        lexeme, originalCode)
}

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

fun expectedException(expected: String, actual: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Expected '${expected}' but found '${actual}'",
    originalCode = originalCode,
    lexeme = lexeme
)

fun expectedException(expected: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Expected '${expected}'",
    originalCode = originalCode,
    lexeme = lexeme
)

fun unexpectedSymbolException(symbol: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Unexpected symbol '${symbol}'",
    originalCode = originalCode,
    lexeme = lexeme
)

fun cannotCastException(what: Type, to: Type, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Cannot cast '${what.text}' to '${to.text}'",
    originalCode = originalCode,
    lexeme = lexeme
)

fun cannotCastException(what: Type, to: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Cannot cast '${what.text}' to '${to}'",
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

fun operatorUsageException(operator: Operator, with: String, lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Operator '${operator.token}' can be only used with $with",
    originalCode = originalCode,
    lexeme = lexeme
)

fun constAssignException(lexeme: Lexeme, originalCode: String) = compilationError(
    text = "Cannot assign new value to constant",
    originalCode = originalCode,
    lexeme = lexeme
)

// For lexer

fun compilationError(text: String, lineIndex: Int, inlineIndex: Int, originalCode: String) = compilationError(
    text = text,
    originalCode = originalCode,
    lexeme = Lexeme("", Lexeme.Type.SPECIAL, lineIndex, inlineIndex)
)

fun unexpectedSymbolInNumberException(symbol: Char, lineIndex: Int, inlineIndex: Int, originalCode: String) = compilationError(
    text = "Unexpected symbol '${symbol}' in number declaration",
    lineIndex = lineIndex,
    inlineIndex = inlineIndex,
    originalCode = originalCode
)

fun tooLargeNumberException(lineIndex: Int, inlineIndex: Int, originalCode: String) = compilationError(
    text = "Too large number",
    lineIndex = lineIndex,
    inlineIndex = inlineIndex,
    originalCode = originalCode
)

fun tooManyFloatingsException(lineIndex: Int, inlineIndex: Int, originalCode: String) = compilationError(
    text = "Too many floating points in number",
    lineIndex = lineIndex,
    inlineIndex = inlineIndex,
    originalCode = originalCode
)