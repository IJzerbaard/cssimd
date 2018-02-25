# CS SIMD

CS SIMD is a failed experiment that would have been an alternative SIMD API in C#.

The idea was to generate the native code by:

- *running* the SIMD intrinsics in an "abstract" sense, building up an intermediate representation
- compiling the intermediate code to native code

While running the SIMD intrinsics abstractly was a cute idea (at least I still think so), it introduces problems that in the end I could not resolve. For example, any control flow in the C# code turns into dynamic code generation, not runtime control flow. So control flow has to be hacked back in. Even assignment in the C# code disappears and has to be hacked back in, which is what finally killed this project.

Perhaps something in here is useful though, so I decided to put it on github anyway.