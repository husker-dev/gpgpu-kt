package com.huskerdev.gpkt.apis.interpreter

import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.apis.interpreter.objects.ExField
import com.huskerdev.gpkt.apis.interpreter.objects.ExScope
import com.huskerdev.gpkt.apis.interpreter.objects.ExValue
import com.huskerdev.gpkt.utils.splitThreadInvocation


class InterpreterProgram(
    val ast: ScopeStatement
): GPProgram(ast) {

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val variables = buffers.associate { field ->
            val desc = when (val value = map[field.name]!!) {
                is CPUAsyncFloatMemoryPointer -> Type.FLOAT_ARRAY to value.array!!
                is CPUAsyncIntMemoryPointer -> Type.INT_ARRAY to value.array!!
                is CPUAsyncByteMemoryPointer -> Type.BYTE_ARRAY to value.array!!
                is CPUSyncFloatMemoryPointer -> Type.FLOAT_ARRAY to value.array!!
                is CPUSyncIntMemoryPointer -> Type.INT_ARRAY to value.array!!
                is CPUSyncByteMemoryPointer -> Type.BYTE_ARRAY to value.array!!
                is Float -> Type.FLOAT to value
                is Int -> Type.INT to value
                is Byte -> Type.BYTE to value
                else -> throw UnsupportedOperationException()
            }
            field.name to ExField(desc.first, ExValue(desc.second))
        }

        splitThreadInvocation(instances) { from, to ->
            val scope = ExScope(ast)
            val iteration = ExField(Type.INT, ExValue(null))
            val threadLocalVariables = hashMapOf(
                "__i__" to iteration
            )
            threadLocalVariables.putAll(variables)

            for(step in from until to){
                iteration.value!!.set(step + indexOffset)
                scope.execute(threadLocalVariables)
            }
        }
    }

    override fun dealloc() = Unit
}

