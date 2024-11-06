package com.huskerdev.gpkt.apis.interpreter

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.apis.interpreter.objects.ExField
import com.huskerdev.gpkt.apis.interpreter.objects.ExScope
import com.huskerdev.gpkt.apis.interpreter.objects.ExValue
import com.huskerdev.gpkt.ast.types.*
import com.huskerdev.gpkt.utils.splitThreadInvocation


class InterpreterProgram(
    val ast: ScopeStatement
): GPProgram(ast) {

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val variables = buffers.associate { field ->
            val desc = when (val value = map[field.name]!!) {
                is CPUAsyncFloatMemoryPointer -> DYNAMIC_FLOAT_ARRAY to value.array!!
                is CPUAsyncIntMemoryPointer -> DYNAMIC_FLOAT_ARRAY to value.array!!
                is CPUAsyncByteMemoryPointer -> DYNAMIC_BYTE_ARRAY to value.array!!
                is CPUSyncFloatMemoryPointer -> DYNAMIC_FLOAT_ARRAY to value.array!!
                is CPUSyncIntMemoryPointer -> DYNAMIC_INT_ARRAY to value.array!!
                is CPUSyncByteMemoryPointer -> DYNAMIC_BYTE_ARRAY to value.array!!
                is Float -> FLOAT to value
                is Int -> INT to value
                is Byte -> BYTE to value
                is Boolean -> BOOLEAN to value
                else -> throw UnsupportedOperationException()
            }
            field.name to ExField(desc.first, ExValue(desc.second))
        }

        splitThreadInvocation(instances) { from, to ->
            val scope = ExScope(ast)
            val iteration = ExField(INT, ExValue(null))
            val threadLocalVariables = hashMapOf(
                "__i__" to iteration
            )
            threadLocalVariables.putAll(variables)

            for(step in from until to){
                iteration.value!!.set(step + indexOffset)
                scope.execute(threadLocalVariables, execMain = true)
            }
        }
    }

    override fun dealloc() = Unit
}

