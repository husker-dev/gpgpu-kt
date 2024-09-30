package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.lexer.Lexeme

class Import(
    val path: String,
    val lexeme: Lexeme
)