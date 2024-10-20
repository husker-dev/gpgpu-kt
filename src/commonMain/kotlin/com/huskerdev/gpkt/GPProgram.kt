package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.types.Type

abstract class GPProgram(ast: ScopeStatement) {
    protected val buffers = ast.scope.fields.filter {
        it.isExtern
    }.toList()

    protected val locals = ast.scope.fields.filter {
        it.isLocal
    }.toList()

    abstract fun dealloc()

    fun executeRange(
        indexOffset: Int,
        instances: Int,
        vararg mapping: Pair<String, Any>
    ) = executeRange(indexOffset, instances, mapOf(*mapping))

    fun execute(
        instances: Int,
        vararg mapping: Pair<String, Any>
    ) = executeRange(0, instances, mapOf(*mapping))

    fun execute(
        instances: Int,
        map: Map<String, Any>
    )  = executeRange(0, instances, map)

    fun executeRange(
        indexOffset: Int,
        instances: Int,
        map: Map<String, Any>
    ) {
        buffers.forEach { field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if (!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)
        }
        executeRangeImpl(indexOffset, instances, map)
    }

    protected abstract fun executeRangeImpl(
        indexOffset: Int,
        instances: Int,
        map: Map<String, Any>
    )

    private fun areEqualTypes(actual: Any, expected: Type): Boolean{
        return expected == when(actual){
            is AsyncFloatMemoryPointer, is SyncFloatMemoryPointer -> Type.FLOAT_ARRAY
            is AsyncIntMemoryPointer, is SyncIntMemoryPointer -> Type.INT_ARRAY
            is AsyncByteMemoryPointer, is SyncByteMemoryPointer -> Type.BYTE_ARRAY
            is Float -> Type.FLOAT
            is Int -> Type.INT
            is Byte -> Type.BYTE
            else -> throw UnsupportedOperationException("Unsupported type: '${actual::class}'")
        }
    }
}

class FieldNotSetException(name: String): Exception("Field '$name' have not been set")

class TypesMismatchException(argument: String): Exception("Value type for argument '$argument' doesn't match.")




