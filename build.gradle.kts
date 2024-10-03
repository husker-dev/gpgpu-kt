@file:OptIn(ExperimentalKotlinGradlePluginApi::class)
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinJsCompile


plugins {
    id("org.jetbrains.kotlin.multiplatform") version "2.1.0-Beta1"
    id("org.jetbrains.kotlinx.benchmark") version "0.4.11"
    id("maven-publish")
}

group = "com.huskerdev"
version = "1.0"

repositories {
    mavenCentral()
}

kotlin {
    jvm {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
        compilations.create("benchmark") {
            associateWith(this@jvm.compilations.getByName("main"))
        }
    }
    js {
        browser()
        binaries.executable()
    }

    sourceSets {
        commonTest.dependencies {
            implementation(kotlin("test"))
        }

        jvmMain {
            dependencies {
                // JOCL (OpenCL)
                /*
                api("org.jocl:jocl:2.0.4")

                // JCuda (Cuda)
                api("org.jcuda:jcuda:12.0.0") {
                    isTransitive = false
                }
                api("org.jcuda:jcuda-natives:12.0.0:windows-x86_64")
                api("org.jcuda:jcuda-natives:12.0.0:linux-x86_64")

                 */


                api(project.dependencies.platform("org.lwjgl:lwjgl-bom:3.3.4"))
                api("org.lwjgl:lwjgl")
                api("org.lwjgl:lwjgl-cuda")
                api("org.lwjgl:lwjgl-opencl")
                api("org.lwjgl:lwjgl-vulkan")

                api("org.lwjgl:lwjgl::natives-windows")
            }
        }

        val jvmBenchmark by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-benchmark-runtime:0.4.11")
            }
        }
    }
}

benchmark {
    targets {
        register("jvmBenchmark")
    }
}

tasks.withType(KotlinJsCompile::class.java).configureEach {
    compilerOptions {
        target = "es2015"
    }
}