package com.huskerdev.gpkt.apis.interpreter.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.allPredefinedFields
import com.huskerdev.gpkt.ast.objects.allPredefinedFunctions
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.Operator.*
import com.huskerdev.gpkt.ast.types.Type
import kotlin.math.*

class BadOperator(operator: Operator): Exception("Can't apply operator '${operator}'")


fun executeExpression(scope: ExScope, expression: Expression): ExValue = when(expression) {
    is AxBExpression -> {
        val left = executeExpression(scope, expression.left)
        val right = executeExpression(scope, expression.right)
        val leftValue = left.get()
        val rightValue = right.get()
        when (val operator = expression.operator) {
            // ==============
            //   Assignment
            // ==============
            ASSIGN -> {
                val castedValue = right.castToType(expression.left.type).get()
                left.set(castedValue)
            }
            PLUS_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue + rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue + rightValue.toInt())
                leftValue is Byte && rightValue is Number -> left.set(leftValue + rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MINUS_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue - rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue - rightValue.toInt())
                leftValue is Byte && rightValue is Number -> left.set(leftValue - rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MULTIPLY_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue * rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue * rightValue.toInt())
                leftValue is Byte && rightValue is Number -> left.set(leftValue * rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            DIVIDE_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue / rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue / rightValue.toInt())
                leftValue is Byte && rightValue is Number -> left.set(leftValue / rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MOD_ASSIGN -> when {
                leftValue is Float && rightValue is Number -> left.set(leftValue % rightValue.toFloat())
                leftValue is Int && rightValue is Number -> left.set(leftValue % rightValue.toInt())
                leftValue is Byte && rightValue is Number -> left.set(leftValue % rightValue.toByte())
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Assignment (Bitwise)
            // ==============
            BITWISE_AND_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue and rightValue.toInt())
                else -> throw BadOperator(operator)
            }
            BITWISE_OR_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue or rightValue.toInt())
                else -> throw BadOperator(operator)
            }
            BITWISE_XOR_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue xor rightValue.toInt())
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_RIGHT_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue shr rightValue.toInt())
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_LEFT_ASSIGN -> when {
                leftValue is Int && rightValue is Number -> left.set(leftValue shl rightValue.toInt())
                else -> throw BadOperator(operator)
            }

            // ==============
            //   Math
            // ==============
            PLUS -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue + rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue + rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue + rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MINUS -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue - rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue - rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue - rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MULTIPLY -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue * rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue * rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue * rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            DIVIDE -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue / rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue / rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue / rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            MOD -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue % rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue % rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue % rightValue.toByte())
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
                leftValue is Int && rightValue is Number -> ExValue(leftValue shr rightValue.toInt())
                else -> throw BadOperator(operator)
            }
            BITWISE_SHIFT_LEFT -> when {
                leftValue is Int && rightValue is Number -> ExValue(leftValue shl rightValue.toInt())
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
                leftValue is Byte && rightValue is Number -> ExValue(leftValue < rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            GREATER -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue > rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue > rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue > rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            LESS_OR_EQUAL -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue <= rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue <= rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue <= rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            GREATER_OR_EQUAL -> when {
                leftValue is Float && rightValue is Number -> ExValue(leftValue >= rightValue.toFloat())
                leftValue is Int && rightValue is Number -> ExValue(leftValue >= rightValue.toInt())
                leftValue is Byte && rightValue is Number -> ExValue(leftValue >= rightValue.toByte())
                else -> throw BadOperator(operator)
            }
            else -> throw UnsupportedOperationException("Unsupported operator '${expression.operator}'")
        }
    }
    is AxExpression -> {
        val left = executeExpression(scope, expression.left)
        val leftValue = left.get()

        // ==============
        //   Increment/decrement
        // ==============
        when (val operator = expression.operator) {
            INCREASE -> when (leftValue) {
                is Float -> left.set(leftValue + 1)
                is Int -> left.set(leftValue + 1)
                is Byte -> left.set(leftValue + 1)
                else -> throw BadOperator(operator)
            }
            DECREASE -> when (leftValue) {
                is Float -> left.set(leftValue - 1)
                is Int -> left.set(leftValue - 1)
                is Byte -> left.set(leftValue - 1)
                else -> throw BadOperator(operator)
            }
            else -> throw UnsupportedOperationException("Unsupported operator '${expression.operator}'")
        }
    }
    is XBExpression -> {
        val right = executeExpression(scope, expression.right)
        val rightValue = right.get()

        // ==============
        //   Math
        // ==============
        when (val operator = expression.operator) {
            POSITIVE -> when (rightValue) {
                is Float, is Int, is Double, is Long, is Byte -> right
                else -> throw BadOperator(operator)
            }
            NEGATIVE -> when (rightValue) {
                is Float -> ExValue(-rightValue)
                is Int -> ExValue(-rightValue)
                is Byte -> ExValue(-rightValue)
                else -> throw BadOperator(operator)
            }
            BITWISE_NOT -> when (rightValue) {
                is Int -> ExValue(rightValue.inv())
                else -> throw BadOperator(operator)
            }
            LOGICAL_NOT -> when (rightValue) {
                is Boolean -> ExValue(!rightValue)
                else -> throw BadOperator(operator)
            }
            else -> throw UnsupportedOperationException("Unsupported operator '${expression.operator}'")
        }
    }
    is ConstExpression -> {
        val text = expression.lexeme.text
        ExValue(
            when (expression.type) {
                Type.FLOAT -> text.toFloat()
                Type.INT -> text.toInt()
                Type.BYTE -> text.toByte()
                Type.BOOLEAN -> text == "true"
                else -> throw UnsupportedOperationException("Can't parse value '${text}'")
            }
        )
    }
    is FunctionCallExpression -> {
        val name = expression.function.name
        if(name !in allPredefinedFunctions) {
            val arguments = expression.function.arguments.mapIndexed { i, arg ->
                arg.name to ExField(arg.type, ExValue(executeExpression(scope, expression.arguments[i]).get()))
            }.toMap().toMutableMap()

            scope.findFunction(expression.function.name)!!.execute(arguments)!!
        }else {
            val values = expression.arguments.map {
                executeExpression(scope, it).get()
            }
            ExValue(when(name){
                "abs" -> abs((values[0] as Number).toFloat())
                "acos" -> acos((values[0] as Number).toFloat())
                "asin" -> asin((values[0] as Number).toFloat())
                "atan" -> atan((values[0] as Number).toFloat())
                "atan2" -> atan2((values[0] as Number).toFloat(), (values[1] as Number).toFloat())
                "cbrt" -> cbrt((values[0] as Number).toFloat())
                "ceil" -> ceil((values[0] as Number).toFloat())
                "cos" -> cos((values[0] as Number).toFloat())
                "cosh" -> cosh((values[0] as Number).toFloat())
                "exp" -> exp((values[0] as Number).toFloat())
                "expm1" -> expm1((values[0] as Number).toFloat())
                "floor" -> floor((values[0] as Number).toFloat())
                "hypot" -> hypot((values[0] as Number).toFloat(), (values[1] as Number).toFloat())
                "log" -> log((values[0] as Number).toFloat(), E.toFloat())
                "log10" -> log10((values[0] as Number).toFloat())
                "max" -> max((values[0] as Number).toFloat(), (values[1] as Number).toFloat())
                "min" -> min((values[0] as Number).toFloat(), (values[1] as Number).toFloat())
                "pow" -> (values[0] as Number).toFloat().pow((values[1] as Number).toInt())
                "round" -> round((values[0] as Number).toFloat())
                "sin" -> sin((values[0] as Number).toFloat())
                "sinh" -> sinh((values[0] as Number).toFloat())
                "sqrt" -> sqrt((values[0] as Number).toFloat())
                "tan" -> tan((values[0] as Number).toFloat())
                "tanh" -> tanh((values[0] as Number).toFloat())
                else -> throw UnsupportedOperationException("Unsupported predefined function")
            })
        }
    }
    is ArrayAccessExpression -> {
        val array = scope.findField(expression.array.field.name)!!.value!!.get()!!
        val index = executeExpression(scope, expression.index).get() as Int
        ExArrayAccessValue(array, index)
    }
    is CastExpression -> executeExpression(scope, expression.right).castToType(expression.type)
    is FieldExpression -> {
        val name = expression.field.name
        if(name !in allPredefinedFields)
            scope.findField(expression.field.name)!!.value!!
        else ExValue(when(name){
            "PI" -> PI.toFloat()
            "E" -> E.toFloat()
            else -> throw UnsupportedOperationException("Unsupported predefined field")
        })
    }
    is BracketExpression -> executeExpression(scope, expression.wrapped)
    else -> throw UnsupportedOperationException("Unsupported expression: $expression")
}
