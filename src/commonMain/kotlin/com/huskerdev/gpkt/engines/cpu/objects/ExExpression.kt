package com.huskerdev.gpkt.engines.cpu.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.Operator.*
import com.huskerdev.gpkt.ast.types.Type

class BadOperator(operator: Operator): Exception("Can't apply operator '${operator}'")


fun executeExpression(scope: ExScope, expression: Expression): ExValue {
    if(expression is AxBExpression){
        val left = executeExpression(scope, expression.left)
        val right = executeExpression(scope, expression.right)
        val leftValue = left.get()
        val rightValue = right.get()
        return when(val operator = expression.operator){
            // ==============
            //   Assignment
            // ==============
            ASSIGN -> left.set(rightValue)
            PLUS_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue + rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue + rightValue.toInt())
                leftValue is Double && rightValue is Number -> left.set(leftValue + rightValue.toDouble())
                leftValue is Long && rightValue is Number -> left.set(leftValue + rightValue.toLong())
                leftValue is Byte && rightValue is Number -> left.set(leftValue + rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MINUS_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue - rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue - rightValue.toInt())
                leftValue is Double && rightValue is Number -> left.set(leftValue - rightValue.toDouble())
                leftValue is Long && rightValue is Number -> left.set(leftValue - rightValue.toLong())
                leftValue is Byte && rightValue is Number -> left.set(leftValue - rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MULTIPLY_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue * rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue * rightValue.toInt())
                leftValue is Double && rightValue is Number -> left.set(leftValue * rightValue.toDouble())
                leftValue is Long && rightValue is Number -> left.set(leftValue * rightValue.toLong())
                leftValue is Byte && rightValue is Number -> left.set(leftValue * rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            DIVIDE_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue / rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue / rightValue.toInt())
                leftValue is Double && rightValue is Number -> left.set(leftValue / rightValue.toDouble())
                leftValue is Long && rightValue is Number -> left.set(leftValue / rightValue.toLong())
                leftValue is Byte && rightValue is Number -> left.set(leftValue / rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MOD_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue % rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue % rightValue.toInt())
                leftValue is Double && rightValue is Number -> left.set(leftValue % rightValue.toDouble())
                leftValue is Long && rightValue is Number -> left.set(leftValue % rightValue.toLong())
                leftValue is Byte && rightValue is Number -> left.set(leftValue % rightValue.toByte())
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Assignment (Bitwise)
            // ==============
            BITWISE_AND_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue and rightValue.toInt())
                leftValue is Long && rightValue is Number -> left.set(leftValue and rightValue.toLong())
                else -> throw BadOperator(operator)
            }
            BITWISE_OR_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue or rightValue.toInt())
                leftValue is Long && rightValue is Number -> left.set(leftValue or rightValue.toLong())
                else -> throw BadOperator(operator)
            }
            BITWISE_XOR_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue xor rightValue.toInt())
                leftValue is Long && rightValue is Number -> left.set(leftValue xor rightValue.toLong())
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_RIGHT_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue shr rightValue.toInt())
                leftValue is Long && rightValue is Number -> left.set(leftValue shr rightValue.toInt())
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_LEFT_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue shl rightValue.toInt())
                leftValue is Long && rightValue is Number -> left.set(leftValue shl rightValue.toInt())
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Math
            // ==============
            PLUS -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue + rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue + rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue + rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue + rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue + rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MINUS -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue - rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue - rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue - rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue - rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue - rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MULTIPLY -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue * rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue * rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue * rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue * rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue * rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            DIVIDE -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue / rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue / rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue / rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue / rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue / rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MOD -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue % rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue % rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue % rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue % rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue % rightValue.toByte())
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Bitwise
            // ==============
            BITWISE_AND -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue and rightValue)
                leftValue is Long && rightValue is Long -> ExValue(leftValue and rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_OR -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue or rightValue)
                leftValue is Long && rightValue is Long -> ExValue(leftValue or rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_XOR -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue xor rightValue)
                leftValue is Long && rightValue is Long -> ExValue(leftValue xor rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_RIGHT -> when {
                leftValue is Int && rightValue is Number -> ExValue(leftValue shr rightValue.toInt())
                leftValue is Long && rightValue is Number -> ExValue(leftValue shr rightValue.toInt())
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_LEFT -> when {
                leftValue is Int && rightValue is Number -> ExValue(leftValue shl rightValue.toInt())
                leftValue is Long && rightValue is Number -> ExValue(leftValue shl rightValue.toInt())
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Logical
            // ==============
            LOGICAL_AND -> when {
                leftValue is Boolean && rightValue is Boolean -> ExValue(leftValue and rightValue)
                else -> throw BadOperator(operator)
            }
            LOGICAL_OR -> when {
                leftValue is Boolean && rightValue is Boolean -> ExValue(leftValue or rightValue)
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Comparison
            // ==============
            EQUAL -> ExValue(leftValue == rightValue)
            NOT_EQUAL -> ExValue(leftValue != rightValue)
            LESS -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue < rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue < rightValue.toFloat())
                leftValue is Double && rightValue is Number -> ExValue(leftValue < rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue < rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue < rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            GREATER -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue > rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue > rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue > rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue > rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue > rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            LESS_OR_EQUAL -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue <= rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue <= rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue <= rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue <= rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue <= rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            GREATER_OR_EQUAL -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue >= rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue >= rightValue.toInt())
                leftValue is Double && rightValue is Number -> ExValue(leftValue >= rightValue.toDouble())
                leftValue is Long && rightValue is Number -> ExValue(leftValue >= rightValue.toLong())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue >= rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            else -> throw UnsupportedOperationException("Unsupported operator '${expression.operator}'")
        }
    }
    if(expression is AxExpression){
        val left = executeExpression(scope, expression.left)
        val leftValue = left.get()

        // ==============
        //   Increment/decrement
        // ==============
        return when(val operator = expression.operator) {
            INCREASE -> when (leftValue) {
                is Float -> left.set(leftValue + 1)
                is Int -> left.set(leftValue + 1)
                is Double -> left.set(leftValue + 1)
                is Long -> left.set(leftValue + 1)
                is Byte -> left.set(leftValue + 1)
                else -> throw BadOperator(operator)
            }
            DECREASE -> when (leftValue) {
                is Float -> left.set(leftValue - 1)
                is Int -> left.set(leftValue - 1)
                is Double -> left.set(leftValue - 1)
                is Long -> left.set(leftValue - 1)
                is Byte -> left.set(leftValue - 1)
                else -> throw BadOperator(operator)
            }
            else -> throw UnsupportedOperationException("Unsupported operator '${expression.operator}'")
        }
    }
    if(expression is XBExpression){
        val right = executeExpression(scope, expression.right)
        val rightValue = right.get()

        // ==============
        //   Math
        // ==============
        return when(val operator = expression.operator) {
            POSITIVE -> when (rightValue) {
                is Float, is Int, is Double, is Long, is Byte -> right
                else -> throw BadOperator(operator)
            }
            NEGATIVE -> when (rightValue) {
                is Float -> ExValue(-rightValue)
                is Int -> ExValue(-rightValue)
                is Double -> ExValue(-rightValue)
                is Long -> ExValue(-rightValue)
                is Byte -> ExValue(-rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_NOT -> when (rightValue) {
                is Int -> ExValue(rightValue.inv())
                is Long -> ExValue(rightValue.inv())
                else -> throw BadOperator(operator)
            }
            LOGICAL_NOT -> when(rightValue) {
                is Boolean -> ExValue(!rightValue)
                else -> throw BadOperator(operator)
            }
            else -> throw UnsupportedOperationException("Unsupported operator '${expression.operator}'")
        }
    }
    if(expression is ConstExpression){
        val text = expression.lexeme.text
        return ExValue(when(expression.type){
            Type.FLOAT -> text.toFloat()
            Type.INT -> text.toInt()
            Type.DOUBLE -> text.toDouble()
            Type.LONG -> text.toLong()
            Type.BYTE -> text.toByte()
            Type.BOOLEAN -> text == "true"
            else -> throw UnsupportedOperationException("Can't parse value '${text}'")
        })
    }
    if(expression is CastExpression){
        val value = executeExpression(scope, expression.right).get()
        val type = expression.type
        return ExValue(when {
            type == Type.FLOAT && value is Number -> value.toFloat()
            type == Type.INT && value is Number -> value.toInt()
            type == Type.DOUBLE && value is Number -> value.toDouble()
            type == Type.LONG && value is Number -> value.toLong()
            type == Type.BYTE && value is Number -> value.toByte()
            else -> throw UnsupportedOperationException("Can't cast '${value}' to '${expression.type.text}'")
        })
    }
    if(expression is FieldExpression) {
        return scope.findField(expression.field.name)!!.value!!
    }
    if(expression is FunctionCallExpression) {
        val arguments = expression.function.arguments.mapIndexed { i, arg ->
            arg.name to ExField(arg.type, ExValue(executeExpression(scope, expression.arguments[i]).get()))
        }.toMap().toMutableMap()
        return scope.findFunction(expression.function.name)!!.execute(arguments)!!
    }
    if(expression is ArrayAccessExpression) {
        val array = scope.findField(expression.array.name)!!.value!!.get()!!
        val index = executeExpression(scope, expression.index).get() as Int
        return ExArrayAccessValue(array, index)
    }
    if(expression is BracketExpression)
        return executeExpression(scope, expression.wrapped)

    throw UnsupportedOperationException("Unsupported expression: $expression")
}
