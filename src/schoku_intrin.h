/*
    Schoku intrinsics

    creates the following intrinsics for x86_64 GNU:
    - bittestandreset, bittestandreset64 (btr instruction)
    - bittestandset, bittestandset64 (bts instruction)
    - bittest, bittest64 (bt instruction)
    - short_popcnt16 (16-bit popcnt)

    all in the Schoku namespace

    These methods are remotely inspired by Windows intrisics of similar names,
    that may also be available under MingW and Cygwin (intrin.h).

    However, they are a little different by name and signature.

    The intrinsics accept a reference to the input operand,
    and the input operand is declared unsigned.
*/

#if defined(__GNUC__) && defined(__x86_64__)

namespace Schoku {

/* This macro is used to create bitset/16/64, bitreset/16/64

Parameters: (FunctionName, DataType, Statement, OffsetConstraint)
   FunctionName: any valid function name
   DataType: __LONG32 or __int64
   Statement: asm statement (btr)
   OffsetConstraint: either "I" for 32bit data types or "J" for 64.
   Type: l, q
*/
#define __buildbitopplus(x, y, z, a, b) bool x(y &Base, y Offset) \
{ \
    bool was_set; \
    __asm__ volatile(z "{" b " %[Offset],%[Base] | %[Base],%[Offset]}" \
        : "=@ccc" (was_set), [Base] "+mr" (Base) \
        : [Offset] a "ri" (Offset) \
        : "cc"); \
    return was_set; \
}

inline __attribute__((always_inline))
__buildbitopplus(bittestandreset, unsigned int, "btr", "I", "l")
inline __attribute__((always_inline))
__buildbitopplus(bittestandreset64, unsigned long long, "btr", "J", "q")

inline __attribute__((always_inline))
__buildbitopplus(bittestandset, unsigned int, "bts", "I", "l")
inline __attribute__((always_inline))
__buildbitopplus(bittestandset64, unsigned long long, "bts", "J", "q")

inline __attribute__((always_inline))
__buildbitopplus(bittest, unsigned int, "bt", "I", "l")
inline __attribute__((always_inline))
__buildbitopplus(bittest, unsigned long long, "bt", "J", "q")

/*
   bittestandcomplement can be easily added.
   Also atomic versions can easily be added using the lock prefix.
*/

}

/*
 * int __builtin_popcnt(x)
 * will accept a short x, but will also always add an expand instruction
 * to return an integer. (at least that's what the assembly output says).
 * 
 * This variant does not generate an extraneous MOVZX instruction.
 */
inline __attribute__((always_inline))
unsigned short short_popcnt16(unsigned short inp)
{
    unsigned short ret;
    __asm__ volatile( " popcntw %w[inp], %[ret]"
        : [ret] "=r" (ret)
        : [inp] "rm" (inp)
        : "cc");
    return ret;
}

#endif
