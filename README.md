1. Create GPGPUEngine
```kotlin
val engine = GPGPUEngine.create()
```

2. Compile a program
```kotlin
val program = engine.compile("""
    in float[] arr1, arr2;
    out float[] result;
    
    void main(int i){
        result[i] = arr1[i] + arr2[i];
    }
""".trimIndent())
```

3. Allocate buffers on GPU
```kotlin
val arr1 = engine.alloc(exampleArray())
val arr2 = engine.alloc(exampleArray())
val result = engine.alloc(arr1.length)

fun exampleArray() = FloatArray(1_000_000) { it.toFloat() }
```

4. Execute
```kotlin
program.execute(
    "arr1" to arr1,
    "arr2" to arr2,
    "result" to result
)
```

5. Read changed buffers
```kotlin
result.read()
```
