package com.huskerdev.gpkt.apis.interpreter.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.allPredefinedFields
import com.huskerdev.gpkt.ast.objects.allPredefinedFunctions
import com.huskerdev.gpkt.ast.types.*
import com.huskerdev.gpkt.ast.types.Operator.*
import kotlin.math.*

class BadOperator(operator: Operator): Exception("Can't apply operator '${operator}'")


fun executeExpression(scope: ExScope, expression: Expression): ExValue {
    return when(expression) {
        is AxBExpression -> {
            val left = executeExpression(scope, expression.left)
            val right = executeExpression(scope, expression.right)
            val leftValue = left.get()
            val rightValue = right.get()
            val type = expression.type
            when (val operator = expression.operator) {
                // ==============
                //   Assignment
                // ==============
                ASSIGN -> {
                    val castedValue = right.castToType(type).get()
                    left.set(castedValue)
                }
                PLUS_ASSIGN -> when {
                    rightValue !is Number -> throw BadOperator(operator)
                    leftValue is Float -> left.set(leftValue + rightValue.toFloat())
                    leftValue is Int -> left.set(leftValue + rightValue.toInt())
                    leftValue is Byte -> left.set(leftValue + rightValue.toByte())
                    else -> throw BadOperator(operator)
                }
                MINUS_ASSIGN -> when {
                    rightValue !is Number -> throw BadOperator(operator)
                    leftValue is Float -> left.set(leftValue - rightValue.toFloat())
                    leftValue is Int -> left.set(leftValue - rightValue.toInt())
                    leftValue is Byte -> left.set(leftValue - rightValue.toByte())
                    else -> throw BadOperator(operator)
                }
                MULTIPLY_ASSIGN -> when {
                    rightValue !is Number -> throw BadOperator(operator)
                    leftValue is Float -> left.set(leftValue * rightValue.toFloat())
                    leftValue is Int -> left.set(leftValue * rightValue.toInt())
                    leftValue is Byte -> left.set(leftValue * rightValue.toByte())
                    else -> throw BadOperator(operator)
                }
                DIVIDE_ASSIGN -> when {
                    rightValue !is Number -> throw BadOperator(operator)
                    leftValue is Float -> left.set(leftValue / rightValue.toFloat())
                    leftValue is Int -> left.set(leftValue / rightValue.toInt())
                    leftValue is Byte -> left.set(leftValue / rightValue.toByte())
                    else -> throw BadOperator(operator)
                }
                MOD_ASSIGN -> when {
                    rightValue !is Number -> throw BadOperator(operator)
                    leftValue is Float -> left.set(leftValue % rightValue.toFloat())
                    leftValue is Int -> left.set(leftValue % rightValue.toInt())
                    leftValue is Byte -> left.set(leftValue % rightValue.toByte())
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
                PLUS -> when(type) {
                    INT -> ExValue(leftValue.asInt() + rightValue.asInt())
                    FLOAT -> ExValue(leftValue.asFloat() + rightValue.asFloat())
                    BYTE -> ExValue(leftValue.asByte() + rightValue.asByte())
                    else -> throw BadOperator(operator)
                }
                MINUS -> when(type) {
                    INT -> ExValue(leftValue.asInt() - rightValue.asInt())
                    FLOAT -> ExValue(leftValue.asFloat() - rightValue.asFloat())
                    BYTE -> ExValue(leftValue.asByte() - rightValue.asByte())
                    else -> throw BadOperator(operator)
                }
                MULTIPLY -> when(type) {
                    INT -> ExValue(leftValue.asInt() * rightValue.asInt())
                    FLOAT -> ExValue(leftValue.asFloat() * rightValue.asFloat())
                    BYTE -> ExValue(leftValue.asByte() * rightValue.asByte())
                    else -> throw BadOperator(operator)
                }
                DIVIDE -> when(type) {
                    INT -> ExValue(leftValue.asInt() / rightValue.asInt())
                    FLOAT -> ExValue(leftValue.asFloat() / rightValue.asFloat())
                    BYTE -> ExValue(leftValue.asByte() / rightValue.asByte())
                    else -> throw BadOperator(operator)
                }
                MOD -> when(type) {
                    INT -> ExValue(leftValue.asInt() % rightValue.asInt())
                    FLOAT -> ExValue(leftValue.asFloat() % rightValue.asFloat())
                    BYTE -> ExValue(leftValue.asByte() % rightValue.asByte())
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
                EQUAL -> {
                    if(leftValue is Number && rightValue is Number)
                        ExValue(leftValue.toFloat() == rightValue.toFloat())
                    else ExValue(leftValue == rightValue)
                }
                NOT_EQUAL -> {
                    if(leftValue is Number && rightValue is Number)
                        ExValue(leftValue.toFloat() != rightValue.toFloat())
                    else ExValue(leftValue != rightValue)
                }
                LESS -> when {
                    leftValue is Number && rightValue is Number ->
                        ExValue(leftValue.toFloat() < rightValue.toFloat())
                    else -> throw BadOperator(operator)
                }
                GREATER -> when {
                    leftValue is Number && rightValue is Number ->
                        ExValue(leftValue.toFloat() > rightValue.toFloat())
                    else -> throw BadOperator(operator)
                }
                LESS_OR_EQUAL -> when {
                    leftValue is Number && rightValue is Number ->
                        ExValue(leftValue.toFloat() <= rightValue.toFloat())
                    else -> throw BadOperator(operator)
                }
                GREATER_OR_EQUAL -> when {
                    leftValue is Number && rightValue is Number ->
                        ExValue(leftValue.toFloat() >= rightValue.toFloat())
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
                    FLOAT -> text.toFloat()
                    INT -> text.toInt()
                    BYTE -> text.toByte()
                    BOOLEAN -> text == "true"
                    else -> throw UnsupportedOperationException("Can't parse value '${text}'")
                }
            )
        }
        is FunctionCallExpression -> {
            val name = expression.function.name

            if(expression.obj != null){
                val objVal = executeExpression(scope, expression.obj).get() as ExClassObject

                val arguments = expression.function.arguments.mapIndexed { i, arg ->
                    arg.name to ExField(arg.type, ExValue(executeExpression(scope, expression.arguments[i]).get()))
                }.toMap().toMutableMap()

                objVal.scope.findFunction(expression.function.name)!!.execute(arguments)
                    ?: ExValue(null)
            }else {
                if (name !in allPredefinedFunctions) {
                    val arguments = expression.function.arguments.mapIndexed { i, arg ->
                        arg.name to ExField(arg.type, ExValue(executeExpression(scope, expression.arguments[i]).get()))
                    }.toMap().toMutableMap()

                    scope.findFunction(name)!!.execute(arguments)
                        ?: ExValue(null)
                } else {
                    val values = expression.arguments.map {
                        executeExpression(scope, it).get()
                    }
                    ExValue(
                        when (name) {
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
                            "pow" -> (values[0] as Number).toFloat().pow((values[1] as Number).toFloat())
                            "round" -> round((values[0] as Number).toFloat())
                            "sin" -> sin((values[0] as Number).toFloat())
                            "sinh" -> sinh((values[0] as Number).toFloat())
                            "sqrt" -> sqrt((values[0] as Number).toFloat())
                            "tan" -> tan((values[0] as Number).toFloat())
                            "tanh" -> tanh((values[0] as Number).toFloat())
                            "isNaN" -> (values[0] as Number).toFloat().isNaN()
                            else -> throw UnsupportedOperationException("Unsupported predefined function '${name}'")
                        }
                    )
                }
            }
        }
        is ArrayAccessExpression -> {
            val arrayExpr = executeExpression(scope, expression.array).get()!!
            val index = (executeExpression(scope, expression.index).get() as Number).toInt()
            ExArrayAccessValue(arrayExpr, index)
        }
        is CastExpression -> executeExpression(scope, expression.right).castToType(expression.type)
        is FieldExpression -> {
            val name = expression.field.name

            if(expression.obj != null){
                val objVal = executeExpression(scope, expression.obj).get() as ExClassObject
                objVal.scope.findField(expression.field.name)!!.value!!
            } else {
                if (name !in allPredefinedFields)
                    scope.findField(name)!!.value!!
                else ExValue(
                    when (name) {
                        "PI" -> PI.toFloat()
                        "E" -> E.toFloat()
                        "NaN" -> Float.NaN
                        else -> throw UnsupportedOperationException("Unsupported predefined field")
                    }
                )
            }
        }
        is ArrayDefinitionExpression -> {
            ExValue(when(expression.type){
                is FloatArrayType -> FloatArray(expression.elements.size)
                is IntArrayType -> IntArray(expression.elements.size)
                is ByteArrayType -> ByteArray(expression.elements.size)
                else -> throw UnsupportedOperationException()
            })
        }
        is ClassCreationExpression -> {
            val instanceScope = ExScope(expression.classObj.body!!.scopeObj, scope)
            val fields = expression.arguments.mapIndexed { i, arg ->
                val name = expression.classObj.variables.keys.toTypedArray()[i]
                name to ExField(arg.type, executeExpression(scope, arg))
            }.toMap().toMutableMap()

            instanceScope.begin(fields = fields)
            expression.classObj.body.scopeObj.statements.forEach {
                instanceScope.evalStatement(it)
            }

            val instance = ExClassObject(
                scope = instanceScope
            )
            ExValue(instance)
        }
        is BracketExpression -> executeExpression(scope, expression.wrapped)
        else -> throw UnsupportedOperationException("Unsupported expression: $expression")
    }
}

private fun Any?.asInt() = when (this) {
    is Int -> this
    is Number -> toInt()
    else -> throw UnsupportedOperationException()
}

private fun Any?.asFloat() = when (this) {
    is Float -> this
    is Number -> toFloat()
    else -> throw UnsupportedOperationException()
}

private fun Any?.asByte() = when (this) {
    is Byte -> this
    is Number -> toByte()
    else -> throw UnsupportedOperationException()
}