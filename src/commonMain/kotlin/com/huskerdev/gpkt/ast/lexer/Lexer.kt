package com.huskerdev.gpkt.ast.lexer

import com.huskerdev.gpkt.ast.*
import kotlin.math.min


fun processLexemes(block: String): List<Lexeme> {
    val lexemes = mutableListOf<Lexeme>()
    val buffer = StringBuilder()
    var inComment = false

    fun parseNumber(text: String, lineIndex: Int, startIndex: Int, block: String): Pair<Lexeme.Type, String>{
        val lowerText = text.lowercase()

        // Check for Hex or Bin
        if(text.length > 2 && text[0] == '0'){
            if(lowerText[1] == 'x'){
                // Check for illegal symbols
                for(i in 2 until lowerText.length)
                    if(!(lowerText[i] in digitsHex || (text[i] == '_' && i > 2 && i < lowerText.length)))
                        throw unexpectedSymbolInNumberException(text[i], lineIndex, startIndex + i, block)

                val actualText = lowerText.substring(2).replace("_", "")
                val actualLength = actualText.length
                return when {
                    actualLength <= 2 -> Lexeme.Type.BYTE to actualText.toByte(16).toString()
                    actualLength <= 8 -> Lexeme.Type.INT to actualText.toLong(16).toInt().toString()
                    actualLength <= 16 -> Lexeme.Type.LONG to actualText.toLong(16).toString()
                    else -> throw tooLargeNumberException(lineIndex, startIndex, block)
                }
            }else if(lowerText[1] == 'b'){
                // Check for illegal symbols
                for(i in 2 until text.length)
                    if(!(text[i] == '0' || text[i] == '1' ||
                        (text[i] == '_' && i > 2 && i < lowerText.length)
                    )) throw unexpectedSymbolInNumberException(text[i], lineIndex, startIndex + i, block)

                val actualText = text.substring(2).replace("_", "")
                val actualLength = actualText.length
                return when {
                    actualLength <= 8 -> Lexeme.Type.BYTE to actualText.toByte(2).toString()
                    actualLength <= 32 -> Lexeme.Type.INT to actualText.toLong(2).toInt().toString()
                    actualLength <= 64 -> Lexeme.Type.LONG to actualText.toLong(2).toString()
                    else -> throw tooLargeNumberException(lineIndex, startIndex, block)
                }
            }
        }

        // Check for Long
        if(lowerText[lowerText.lastIndex] == 'l') {
            // Check for illegal symbols
            lowerText.forEachIndexed { i, char ->
                if(!(char in digits ||
                    (char == '_' && i > 0 && i < lowerText.length-2) ||
                    (char == 'l' && i == lowerText.lastIndex)
                )) throw unexpectedSymbolInNumberException(text[i], lineIndex, startIndex + i, block)
            }
            throw compilationError("Long is not supported :(", lineIndex, startIndex, block)
            //return Lexeme.Type.LONG to text.replace("_", "").substring(0, text.lastIndex)
        }

        // Check for Float
        if(lowerText[lowerText.lastIndex] == 'f') {
            if("." in text && text.length == 2)
                throw unexpectedSymbolInNumberException(text[0], lineIndex, startIndex, block)
            // Check for illegal symbols
            lowerText.forEachIndexed { i, char ->
                if(!(char in digits ||
                    (char == '_' && i > 0 && i < lowerText.length-2) ||
                    char == '.' ||
                    (char == 'f' && i == lowerText.lastIndex)
                )) throw unexpectedSymbolInNumberException(text[i], lineIndex, startIndex + i, block)
            }
            // Check for floating points
            if(text.indexOf('.') != text.lastIndexOf('.'))
                throw tooManyFloatingsException(lineIndex, startIndex + text.lastIndexOf('.'), block)
            return Lexeme.Type.FLOAT to text.replace("_", "").substring(0, text.lastIndex)
        }

        // Check for Double
        if("." in text){
            if(text.length == 1)
                throw unexpectedSymbolInNumberException(text[0], lineIndex, startIndex, block)
            // Check for illegal symbols
            lowerText.forEachIndexed { i, char ->
                if(!(char in digits ||
                    char == '.' ||
                    (char == '_' && i > 0 && i < lowerText.length-1)
                )) throw unexpectedSymbolInNumberException(text[i], lineIndex, startIndex + i, block)
            }
            if(text.indexOf('.') != text.lastIndexOf('.'))
                throw tooManyFloatingsException(lineIndex, startIndex + text.lastIndexOf('.'), block)
            return Lexeme.Type.DOUBLE to text.replace("_", "")
        }

        // > Check for Int
        // Check for illegal symbols
        lowerText.forEachIndexed { i, char ->
            if (!(char in digits ||
                (char == '_' && i > 0 && i < lowerText.length)
            )) throw unexpectedSymbolInNumberException(text[i], lineIndex, startIndex + i, block)
        }
        return Lexeme.Type.INT to text.replace("_", "")
    }

    fun flush(lineIndex: Int, charIndex: Int) {
        if(buffer.isEmpty())
            return
        var text = buffer.toString()
        val startIndex = charIndex - text.length
        buffer.setLength(0)

        val type = if(text[0] in digits || text[0] == '.'){
            val result = parseNumber(text, lineIndex, startIndex, block)
            text = result.second
            result.first
        } else if(text in logical)
            Lexeme.Type.LOGICAL
        else if(text in specials_keywords)
            Lexeme.Type.SPECIAL
        else Lexeme.Type.NAME

        lexemes.add(Lexeme(text, type, lineIndex, startIndex))
    }

    try {
        block.split("\n").forEachIndexed { lineIndex, line ->
            var i = 0
            while (i < line.length) {
                val char = line[i]

                if (char == '/') {
                    if (!inComment && line[i + 1] == '/') { // Line comment
                        flush(lineIndex, i)
                        i = line.length
                        continue
                    } else if (!inComment && line[i + 1] == '*') { // Multi-line comment begin
                        flush(lineIndex, i)
                        inComment = true
                        i += 2
                        continue
                    } else if (inComment && line[i - 1] == '*') { // Multi-line comment end
                        inComment = false
                        i += 2
                        continue
                    }
                }
                if (inComment) {
                    i++
                    continue
                }

                // Trying to get space
                if (char in spacing) {
                    flush(lineIndex, i)
                    i++
                    continue
                }

                // Trying to get special operator
                var foundSpecialOp = false
                for (r in min(line.length, i + longestSpecial) downTo i) {
                    val text = line.substring(i, r)
                    if (text in specials_separators) {
                        flush(lineIndex, i)
                        lexemes.add(Lexeme(text, Lexeme.Type.SPECIAL, lineIndex, i))
                        i = r
                        foundSpecialOp = true
                        break
                    }
                }
                if (foundSpecialOp)
                    continue

                // Append buffer
                buffer.append(char)
                i++
            }
            flush(lineIndex, i)
        }
    }catch (e: IndexOutOfBoundsException){
        throw unexpectedEofException(lexemes.last(), block)
    }
    return lexemes
}