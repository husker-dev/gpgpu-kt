package com.huskerdev.gpkt.apis.interpreter.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.Modifiers


class ExScope(
    val scope: GPScope?,
    val parentScope: ExScope? = null
) {
    private var began = false
    var fields: MutableMap<String, ExField>? = null
    private var functions: MutableMap<String, ExScope>? = null

    fun execute(
        fields: MutableMap<String, ExField> = hashMapOf(),
        functions: MutableMap<String, ExScope> = hashMapOf(),
        execMain: Boolean = false
    ): ExValue? {
        begin(fields, functions)
        scope?.statements?.forEach { statement ->
            evalStatement(statement)?.apply {
                end()
                return this
            }
        }
        if(execMain){
            val mainFunc = scope!!.functions["main"]!!
            val mainFuncEx = functions["main"]!!

            mainFuncEx.execute(hashMapOf(mainFunc.arguments[0].name to fields["__i__"]!!))
        }
        end()
        return null
    }

    fun begin(
        fields: MutableMap<String, ExField> = hashMapOf(),
        functions: MutableMap<String, ExScope> = hashMapOf(),
    ){
        if(began) return
        began = true
        this.fields = fields
        this.functions = functions
    }

    fun end(){
        if(!began) return
        began = false
        this.fields = null
        this.functions = null
    }

    fun evalStatement(it: Statement): ExValue? {
        when(it) {
            is FieldStatement -> it.fields.forEach { field ->
                if(Modifiers.EXTERNAL !in field.modifiers) {
                    val value = if (field.initialExpression != null)
                        executeExpression(this, field.initialExpression!!).castToType(field.type)
                    else null
                    addField(field.name, ExField(field.type, value))
                }
            }
            is FunctionStatement -> {
                if(it !is FunctionDefinitionStatement)
                    addFunction(it.function.name, ExScope(it.function.body!!.scopeObj, this))
            }
            is ExpressionStatement -> executeExpression(this, it.expression)
            is ReturnStatement -> return if(it.expression != null)
                executeExpression(this, it.expression)
            else null
            is IfStatement -> {
                if(executeExpression(this, it.condition).get() == true)
                    evalStatement(it.body)?.let { return it }
                else if(it.elseBody != null)
                    evalStatement(it.elseBody)?.let { return it }
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
            is ScopeStatement -> ExScope(it.scopeObj, this).execute()
            is BreakStatement -> return BreakMarker
            is ContinueStatement -> return ContinueMarker
            is ClassStatement, is EmptyStatement -> Unit
            else -> throw UnsupportedOperationException("Unsupported statement '$it'")
        }
        return null
    }

    private fun addField(name: String, field: ExField){
        fields!![name] = field
    }

    private fun addFunction(name: String, scope: ExScope){
        functions!![name] = scope
    }

    fun findField(name: String): ExField? =
        fields!![name] ?: parentScope?.findField(name)

    fun findFunction(name: String): ExScope? =
        functions!![name] ?: parentScope?.findFunction(name)
}