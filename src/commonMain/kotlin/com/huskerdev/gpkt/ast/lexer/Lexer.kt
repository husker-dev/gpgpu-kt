package com.huskerdev.gpkt.ast.lexer

import com.huskerdev.gpkt.ast.compilationError
import com.huskerdev.gpkt.ast.unexpectedEofException
import kotlin.math.min


fun processLexemes(block: String): List<Lexeme> {
    val lexemes = mutableListOf<Lexeme>()
    val buffer = StringBuilder()
    var inComment = false

    fun flush(lineIndex: Int, charIndex: Int) {
        if(buffer.isEmpty())
            return
        var text = buffer.toString()
        val startIndex = charIndex - text.length
        buffer.setLength(0)

        val type = if(text[0] in digits){
            // Parse as a number
            text.forEachIndexed { i, c ->
                if(c != '.' && c != '_' && c !in digits)
                    throw compilationError("illegal symbol '${c}' in number declaration", lineIndex, startIndex + i, block)
            }
            if(text.indexOf('.') == text.lastIndex)
                throw compilationError("unfinished floating-point expression", lineIndex, startIndex + text.indexOf('.'), block)
            if(text.indexOf('_') == text.lastIndex)
                throw compilationError("illegal underscore at the end of the number", lineIndex, startIndex + text.indexOf('_'), block)
            if(text.lastIndexOf('.') != text.indexOf('.'))
                throw compilationError("too many floating points", lineIndex, startIndex + text.lastIndexOf('.'), block)

            text = text.replace("_", "")
            if("." in text) Lexeme.Type.NUMBER_FLOATING_POINT else Lexeme.Type.NUMBER
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