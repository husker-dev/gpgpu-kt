package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.types.*

abstract class GPProgram(ast: ScopeStatement) {
    protected val buffers = ast.scope.fields.filter {
        it.value.isExtern
    }.map { it.value }.toList()

    protected val locals = ast.scope.fields.filter {
        it.value.isLocal
    }.map { it.value }.toList()

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

    private fun areEqualTypes(actual: Any, expected: PrimitiveType): Boolean{
        return when(actual){
            is AsyncFloatMemoryPointer, is SyncFloatMemoryPointer -> expected is FloatArrayType
            is AsyncIntMemoryPointer, is SyncIntMemoryPointer -> expected is IntArrayType
            is AsyncByteMemoryPointer, is SyncByteMemoryPointer -> expected is ByteArrayType
            is Float -> expected is FloatType
            is Int -> expected is IntType
            is Byte -> expected is ByteType
            else -> throw UnsupportedOperationException("Unsupported type: '${actual::class}'")
        }
    }
}

class FieldNotSetException(name: String): Exception("Field '$name' have not been set")

class TypesMismatchException(argument: String): Exception("Value type for argument '$argument' doesn't match.")




