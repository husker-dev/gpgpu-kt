package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPClass
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.*
import com.huskerdev.gpkt.utils.Dictionary


fun parseClassStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
): ClassStatement {
    var i = from + 1

    // Name
    val nameLexeme = lexemes[i]
    if(nameLexeme.type != Lexeme.Type.NAME)
        throw expectedException("class name", nameLexeme, codeBlock)
    val name = nameLexeme.text
    val obfName = dictionary.nextWord()
    i += 1

    // Variables
    val variables = linkedMapOf<String, GPField>()
    val variablesTypes = ArrayList<PrimitiveType>()
    if(lexemes[i].text != "(")
        throw expectedException("class variables", lexemes[i], codeBlock)
    i++
    while(lexemes[i].text != ")"){
        if(i >= to)
            throw compilationError("Expected ')'", lexemes.last(), codeBlock)

        val fieldDeclaration = parseFieldDeclaration(
            scope,
            lexemes,
            codeBlock,
            i,
            to,
            dictionary,
            allowMultipleDeclaration = false,
            allowDefaultValue = false,
            endsWithSemicolon = false
        )
        val field = fieldDeclaration.fields[0]
        variables[field.name] = field
        variablesTypes += field.type

        i += fieldDeclaration.lexemeLength
        if(lexemes[i].text == ",")
            i++
    }
    i++

    // Inheritance
    val implements = arrayListOf<String>()
    if(lexemes[i].text == ":"){
        i++
        while(i < to){
            val l = lexemes[i]
            if(l.type != Lexeme.Type.NAME)
                throw expectedException("class name", nameLexeme, codeBlock)
            if(l.text !in setOf("Float", "Int", "Byte", "Boolean"))
                throw expectedException("available parent types are: [Float, Int, Byte, Boolean]", nameLexeme, codeBlock)
            implements += l.text
            i++

            val nextL = lexemes[i]
            if(nextL.text == ",")
                i += 2
            else break
        }
    }

    // Type
    val type = object: ClassType {
        override val className = name
        override val toArray = { _: Int -> throw UnsupportedOperationException() }
        override val toDynamicArray = { throw UnsupportedOperationException() }
        override val bytes = when {
            "Int" in implements || "Float" in implements -> 4
            "Boolean" in implements || "Byte" in implements -> 1
            else -> 0
        }
        override val isFloating = "Float" in implements
        override val isInteger = "Int" in implements || "Byte" in implements
        override val isLogical = "Boolean" in implements
        override val isArray = false
        override val isDynamicArray = false
        override val isConstArray = false
        override fun toString() = obfName
    }

    // Body
    var bodyScopeStatement: ScopeStatement? = null

    if(lexemes[i].text == "{") {
        bodyScopeStatement =
            parseScopeStatement(scope, lexemes, codeBlock, i + 1, to, dictionary, fields = variables)
        val bodyScope = bodyScopeStatement.scope
        i++

        // Check get/set if inherited from primitive
        mapOf(
            "Float" to FLOAT,
            "Int" to INT,
            "Byte" to BYTE,
            "Boolean" to BOOLEAN
        ).forEach { e ->
            if (e.key !in implements)
                return@forEach
            val className = e.key
            val primitive: SinglePrimitiveType<*> = e.value

            // Getter
            val getFunc = bodyScope.functions["get$className"]
                ?: throw functionNotImplementedInClass("$primitive get$className(){..}", lexemes[i], codeBlock)
            if (getFunc.returnType != primitive)
                throw compilationError("'get$className' should have type '$primitive'", lexemes[i], codeBlock)
            if (getFunc.arguments.isNotEmpty())
                throw compilationError("'get$className' should not have arguments", lexemes[i], codeBlock)

            // Setter
            val setFunc = bodyScope.functions["set$className"]
                ?: throw functionNotImplementedInClass("void set$className($primitive){..}", lexemes[i], codeBlock)
            if (setFunc.returnType != VOID)
                throw compilationError("'set$className' should have type 'void'", lexemes[i], codeBlock)
            if (setFunc.argumentsTypes.size != 1 && setFunc.argumentsTypes[0] != primitive)
                throw compilationError("'set$className' should have one '$primitive' argument", lexemes[i], codeBlock)
        }
        i += bodyScopeStatement.lexemeLength
    }

    val classObj = GPClass(scope, name, type, variables, variablesTypes, bodyScopeStatement, obfName, implements)

    return ClassStatement(scope, classObj, from, i - from)
}