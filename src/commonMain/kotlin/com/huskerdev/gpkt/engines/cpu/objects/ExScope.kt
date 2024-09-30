package com.huskerdev.gpkt.engines.cpu.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.types.Modifiers


class ExScope(
    val scope: ScopeStatement?,
    private val parentScope: ExScope? = null
) {
    private var began = false
    private var fields: MutableMap<String, ExField>? = null
    private var functions: MutableMap<String, ExScope>? = null

    fun execute(
        fields: MutableMap<String, ExField> = hashMapOf(),
        functions: MutableMap<String, ExScope> = hashMapOf(),
    ): ExValue? {
        begin(fields, functions)
        scope?.statements?.forEach { statement ->
            evalStatement(statement)?.apply {
                end()
                return this
            }
        }
        end()
        return null
    }

    private fun begin(
        fields: MutableMap<String, ExField> = hashMapOf(),
        functions: MutableMap<String, ExScope> = hashMapOf(),
    ){
        if(began) return
        began = true
        this.fields = fields
        this.functions = functions
    }

    private fun end(){
        if(!began) return
        began = false
        this.fields = null
        this.functions = null
    }

    private fun evalStatement(it: Statement): ExValue? {
        when(it) {
            is FieldStatement -> it.fields.forEach { field ->
                if(Modifiers.EXTERNAL !in field.modifiers) {
                    addField(field.name, ExField(field.type,
                        if (field.initialExpression != null)
                            executeExpression(this, field.initialExpression).castToType(field.type)
                        else null
                    ))
                }
            }
            is FunctionStatement -> addFunction(it.function.name, ExScope(it.function.body, this))
            is ExpressionStatement -> executeExpression(this, it.expression)
            is ReturnStatement -> return if(it.expression != null)
                executeExpression(this, it.expression).castToType(scope!!.scope.returnType)
            else null
            is IfStatement -> {
                if(executeExpression(this, it.condition).get() == true)
                    evalStatement(it.body)
                else if(it.elseBody != null)
                    evalStatement(it.elseBody)
            }
            is WhileStatement -> {
                while(executeExpression(this, it.condition).get() == true){
                    val res = evalStatement(it.body)
                    if(res != null) {
                        if (res == BreakMarker) break
                        if (res == ContinueMarker) continue
                        return res
                    }
                }
            }
            is ForStatement -> {
                val forScope = ExScope(null, this)
                forScope.begin()
                forScope.evalStatement(it.initialization)

                val condition = it.condition
                while(condition == null || executeExpression(forScope, condition).get() == true){
                    val res = forScope.evalStatement(it.body)
                    if(res != null) {
                        if (res == BreakMarker) break
                        if (res == ContinueMarker) continue
                        return res
                    }
                    if(it.iteration != null)
                        executeExpression(forScope, it.iteration)
                }
                forScope.end()
            }
            is BreakStatement -> return BreakMarker
            is ContinueStatement -> return ContinueMarker
            is ImportStatement -> Unit
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
            scope.execute(hashMapOf("i" to fields!!["__i__"]!!))
    }

    fun findField(name: String): ExField? =
        fields!![name] ?: parentScope?.findField(name)

    fun findFunction(name: String): ExScope? =
        functions!![name] ?: parentScope?.findFunction(name)
}