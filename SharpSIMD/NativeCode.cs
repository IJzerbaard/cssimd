using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace SharpSIMD
{
    class NativeCode : IDisposable
    {
        IntPtr nativemem;
        UIntPtr size;

        public NativeCode(List<byte> contents)
        {
#if DEBUG
            var sw = System.Diagnostics.Stopwatch.StartNew();
#endif
            const int alignment = 1 << 12;
            size = (UIntPtr)((contents.Count + alignment - 1) & -alignment);
            nativemem = VirtualAlloc(IntPtr.Zero, size, AllocationType.COMMIT | AllocationType.RESERVE, MemoryProtection.READWRITE);
            if (nativemem == IntPtr.Zero)
                throw new Exception("Allocation failure");
            Marshal.Copy(contents.ToArray(), 0, nativemem, contents.Count);
            MemoryProtection ignore;
            if (!VirtualProtect(nativemem, size, MemoryProtection.EXECUTE_READ, out ignore))
                throw new Exception("Could not make memory executable");
#if DEBUG
            sw.Stop();
            System.Diagnostics.Debug.WriteLine("Writing code to unmanaged memory: {0}ms", sw.ElapsedMilliseconds);
#endif
        }

        public System.Delegate GetDelegate(Type T, int offset)
        {
            return Marshal.GetDelegateForFunctionPointer(nativemem + offset, T);
        }

        public void Dispose()
        {
            if (nativemem != IntPtr.Zero)
                VirtualFree(nativemem, size, 0x8000);
            GC.SuppressFinalize(this);
        }

        ~NativeCode()
        {
            if (nativemem != IntPtr.Zero)
                VirtualFree(nativemem, size, 0x8000);
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr VirtualAlloc(IntPtr lpAddress, UIntPtr dwSize, AllocationType lAllocationType, MemoryProtection flProtect);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool VirtualProtect(IntPtr lpAddress, UIntPtr dwSize, MemoryProtection flNewProtect, out MemoryProtection lpflOldProtect);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool VirtualFree(IntPtr lpAddress, UIntPtr dwSize, uint dwFreeType);

        [Flags]
        public enum AllocationType : uint
        {
            COMMIT = 0x1000,
            RESERVE = 0x2000,
            RESET = 0x80000,
            LARGE_PAGES = 0x20000000,
            PHYSICAL = 0x400000,
            TOP_DOWN = 0x100000,
            WRITE_WATCH = 0x200000
        }

        [Flags]
        public enum MemoryProtection : uint
        {
            EXECUTE = 0x10,
            EXECUTE_READ = 0x20,
            EXECUTE_READWRITE = 0x40,
            EXECUTE_WRITECOPY = 0x80,
            NOACCESS = 0x01,
            READONLY = 0x02,
            READWRITE = 0x04,
            WRITECOPY = 0x08,
            GUARD_Modifierflag = 0x100,
            NOCACHE_Modifierflag = 0x200,
            WRITECOMBINE_Modifierflag = 0x400
        }
    }
}
