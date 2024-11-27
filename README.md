# gpgpu-kt
Cross-platform general-purpose computing Kotlin Multiplatform library

### Available accelerations:
|         | OpenCL             | CUDA               | Vulkan | Metal                | WebGPU    | JS                 | JVM                |
|---------|--------------------|--------------------|--------|----------------------|-----------|--------------------|--------------------|
| jvm     | :white_check_mark: | :white_check_mark: | :x:    |  :white_check_mark:  |           |                    | :white_check_mark: |
| js      |                    |                    |        |                      |:warning:* | :white_check_mark: |                    |
| wasm    |                    |                    |        |                      |    :x:    |         :x:        |                    |
| ios     |                    |                    |        |  :white_check_mark:  |           |                    |                    |
| android | :white_check_mark: |                    | :x:    |                      |           |                    |  :x:               |
| macos   | :x:                |                    |        |  :white_check_mark:  |           |                    |                    |
| windows | :x:                | :white_check_mark: | :x:    |                      |           |                    |                    |
| linux   | :x:                | :white_check_mark: | :x:    |                      |           |                    |                    |

- :white_check_mark: - Fully implemented
- :warning: - Partially working
- :x: - Not implemented, but planned

- \* Doesn't support loops and Byte 

### TODO List:
  - [ ] Add `sizeof()` function
  - [ ] Add Vulkan support

## Documentation
Read [Wiki](../../wiki)
