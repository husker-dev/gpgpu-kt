package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.lexer.Lexeme

class Import(
    val paths: List<String>,
    val lexeme: Lexeme
)