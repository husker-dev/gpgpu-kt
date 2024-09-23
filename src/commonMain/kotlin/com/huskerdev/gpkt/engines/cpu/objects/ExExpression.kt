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
                leftValue is Float && rightValue is Float -> left.set(leftValue + rightValue)
                leftValue is Int && rightValue is Int -> left.set(leftValue + rightValue)
                else -> throw BadOperator(operator)
            }
            MINUS_ASSIGN -> when {
                leftValue is Float && rightValue is Float -> left.set(leftValue - rightValue)
                leftValue is Int && rightValue is Int -> left.set(leftValue - rightValue)
                else -> throw BadOperator(operator)
            }
            MULTIPLY_ASSIGN -> when {
                leftValue is Float && rightValue is Float -> left.set(leftValue * rightValue)
                leftValue is Int && rightValue is Int -> left.set(leftValue * rightValue)
                else -> throw BadOperator(operator)
            }
            DIVIDE_ASSIGN -> when {
                leftValue is Float && rightValue is Float -> left.set(leftValue / rightValue)
                leftValue is Int && rightValue is Int -> left.set(leftValue / rightValue)
                else -> throw BadOperator(operator)
            }
            MOD_ASSIGN -> when {
                leftValue is Float && rightValue is Float -> left.set(leftValue % rightValue)
                leftValue is Int && rightValue is Int -> left.set(leftValue % rightValue)
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Assignment (Bitwise)
            // ==============
            BITWISE_AND_ASSIGN -> when {
                leftValue is Int && rightValue is Int -> left.set(leftValue and rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_OR_ASSIGN -> when {
                leftValue is Int && rightValue is Int -> left.set(leftValue or rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_XOR_ASSIGN -> when {
                leftValue is Int && rightValue is Int -> left.set(leftValue xor rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_RIGHT_ASSIGN -> when {
                leftValue is Int && rightValue is Int -> left.set(leftValue shr rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_LEFT_ASSIGN -> when {
                leftValue is Int && rightValue is Int -> left.set(leftValue shl rightValue)
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Math
            // ==============
            PLUS -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue + rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue + rightValue)
                else -> throw BadOperator(operator)
            }
            MINUS -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue - rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue - rightValue)
                else -> throw BadOperator(operator)
            }
            MULTIPLY -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue * rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue * rightValue)
                else -> throw BadOperator(operator)
            }
            DIVIDE -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue / rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue / rightValue)
                else -> throw BadOperator(operator)
            }
            MOD -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue % rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue % rightValue)
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Bitwise
            // ==============
            BITWISE_AND -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue and rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_OR -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue or rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_XOR -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue xor rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_RIGHT -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue shr rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_LEFT -> when {
                leftValue is Int && rightValue is Int -> ExValue(leftValue shl rightValue)
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
                leftValue is Float && rightValue is Float -> ExValue(leftValue < rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue < rightValue)
                else -> throw BadOperator(operator)
            }
            GREATER -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue > rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue > rightValue)
                else -> throw BadOperator(operator)
            }
            LESS_OR_EQUAL -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue <= rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue <= rightValue)
                else -> throw BadOperator(operator)
            }
            GREATER_OR_EQUAL -> when {
                leftValue is Float && rightValue is Float -> ExValue(leftValue >= rightValue)
                leftValue is Int && rightValue is Int -> ExValue(leftValue >= rightValue)
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
                else -> throw BadOperator(operator)
            }
            DECREASE -> when (leftValue) {
                is Float -> left.set(leftValue - 1)
                is Int -> left.set(leftValue - 1)
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
                is Float -> right
                is Int -> right
                else -> throw BadOperator(operator)
            }
            NEGATIVE -> when (rightValue) {
                is Float -> ExValue(-rightValue)
                is Int -> ExValue(-rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_NOT -> when {
                rightValue is Int -> ExValue(rightValue.inv())
                else -> throw BadOperator(operator)
            }
            LOGICAL_NOT -> when {
                rightValue is Boolean -> ExValue(!rightValue)
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
            Type.BOOLEAN -> text == "true"
            else -> throw UnsupportedOperationException("Can't parse value '${text}'")
        })
    }
    if(expression is CastExpression){
        val value = executeExpression(scope, expression.right).get()
        val type = expression.type
        return ExValue(when {
            type == Type.FLOAT && value is Int -> value.toFloat()
            type == Type.INT && value is Float -> value.toInt()
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
