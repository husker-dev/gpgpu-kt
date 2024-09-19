package com.huskerdev.gpkt.engines.cpu.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Modifiers

class ExScope(
    val scope: Scope,
    private val parentScope: ExScope? = null
) {
    private var fields: MutableMap<String, ExField>? = null
    private var functions: MutableMap<String, ExScope>? = null

    fun execute(
        fields: MutableMap<String, ExField> = hashMapOf(),
        functions: MutableMap<String, ExScope> = hashMapOf(),
    ): ExValue? {
        this.fields = fields
        this.functions = functions
        scope.statements.forEach { statement ->
            evalStatement(statement)?.apply { return this }
        }
        this.fields = null
        this.functions = null
        return null
    }

    private fun evalStatement(it: Statement): ExValue? {
        when(it) {
            is FieldStatement -> it.fields.forEach { field ->
                if(Modifiers.IN !in field.modifiers && Modifiers.OUT !in field.modifiers) {
                    addField(field.name, ExField(field.type,
                        if (field.initialExpression != null)
                            executeExpression(this, field.initialExpression)
                        else null
                    ))
                }
            }
            is FunctionStatement -> addFunction(it.function.name, ExScope(it.function, this))
            is ExpressionStatement -> executeExpression(this, it.expression)
            is ReturnStatement -> return if(it.expression != null)
                executeExpression(this, it.expression)
            else null
            is IfStatement -> {
                if(executeExpression(this, it.condition).get() == true)
                    ExScope(it.body, this).execute()?.apply { return this }
                else if(it.elseBody != null)
                    ExScope(it.elseBody, this).execute()?.apply { return this }
            }
            is WhileStatement -> {
                while(executeExpression(this, it.condition).get() == true){
                    ExScope(it.body, this).execute()?.apply { return this }
                }
            }
            is ForStatement -> {
                val scope = ExScope(it.body, this)
                scope.evalStatement(it.initialization)

                val condition = it.condition
                val body = ExScope(scope.scope, scope)

                while(condition !is ExpressionStatement || executeExpression(scope, condition.expression).get() == true){
                    body.execute()?.apply { return this }
                    scope.evalStatement(it.iteration)
                }
            }
            else -> throw UnsupportedOperationException("Unsupported statement '$it'")
        }
        return null
    }

    private fun addField(name: String, field: ExField){
        fields!![name] = field
    }

    private fun addFunction(name: String, scope: ExScope){
        functions!![name] = scope
        if(name == "main")
            scope.execute(hashMapOf("i" to fields!!["__i"]!!))
    }

    fun findField(name: String): ExField? =
        fields!![name] ?: parentScope?.findField(name)

    fun findFunction(name: String): ExScope? =
        functions!![name] ?: parentScope?.findFunction(name)
}