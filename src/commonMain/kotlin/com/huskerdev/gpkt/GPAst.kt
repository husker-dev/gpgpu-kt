package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.ast.lexer.processLexemes
import com.huskerdev.gpkt.ast.parser.parseScope


class GPAst {
    companion object {
        fun parse(text: String): Scope {
            val lexemes = processLexemes(text)
            //println(lexemes.mapIndexed { index, lexeme -> "$index:${lexeme.text}" }.joinToString(" "))

            val program = Scope(null, Type.VOID)
            parseScope(program, lexemes, text, 0, lexemes.size)
            return program
        }
    }
}

