package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.lexer.processLexemes
import com.huskerdev.gpkt.ast.parser.parseScopeStatement


class GPAst {
    companion object {
        fun parse(text: String, device: GPDeviceBase? = null): ScopeStatement {
            val lexemes = processLexemes(text)
            //println(lexemes.mapIndexed { index, lexeme -> "$index:${lexeme.text}" }.joinToString(" "))

            return parseScopeStatement(
                parentScope = null,
                lexemes = lexemes,
                codeBlock = text,
                from = 0,
                to = lexemes.size,
                device = device,
            )
        }
    }
}

