@file:OptIn(ExperimentalKotlinGradlePluginApi::class)
import org.jetbrains.kotlin.gradle.ExperimentalKotlinGradlePluginApi
import org.jetbrains.kotlin.gradle.dsl.JvmTarget


plugins {
    id("org.jetbrains.kotlin.multiplatform").version("2.0.20")
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
    }
    js {
        browser()
    }

    sourceSets {
        commonTest.dependencies {
            implementation(kotlin("test"))
        }

        jvmMain {
            dependencies {
                // JOCL (OpenCL)
                api("org.jocl:jocl:2.0.4")

                // JCuda (Cuda)
                api("org.jcuda:jcuda:12.0.0"){
                    isTransitive = false
                }
                api("org.jcuda:jcuda-natives:12.0.0:windows-x86_64")
                api("org.jcuda:jcuda-natives:12.0.0:linux-x86_64")
            }
        }
    }
}