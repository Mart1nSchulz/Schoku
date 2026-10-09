/*
    Schoku intrinsics

    creates the following intrinsics for x86_64 GNUC:
    - _btr_u32/64, (bittestandreset) btr instruction
    - _bts_u32/64, (bittestandset) bts instruction
    - _bt_u32/64, (bittest) bt instruction
    - _btc_u32/64, (bittestandcomplement) btc instruction
    - _popcnt_u16, 16-bit popcnt
    - _tzcnt_u16
    - _lzcnt_u16

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
    __asm__ volatile(z "{" b "\t%[Offset],%[Base] | %[Base],%[Offset]}" \
        : "=@ccc" (was_set), [Base] "+mr" (Base) \
        : [Offset] a "ri" (Offset) \
        : "cc"); \
    return was_set; \
}

inline __attribute__((always_inline))
__buildbitopplus(_btr_u32, unsigned int, "btr", "I", "l")
inline __attribute__((always_inline))
__buildbitopplus(_btr_u64, unsigned long long, "btr", "J", "q")

inline __attribute__((always_inline))
__buildbitopplus(_bts_u32, unsigned int, "bts", "I", "l")
inline __attribute__((always_inline))
__buildbitopplus(_bts_u64, unsigned long long, "bts", "J", "q")

inline __attribute__((always_inline))
__buildbitopplus(_bt_u32, unsigned int, "bt", "I", "l")
inline __attribute__((always_inline))
__buildbitopplus(_bt_u64, unsigned long long, "bt", "J", "q")

inline __attribute__((always_inline))
__buildbitopplus(_btc_u32, unsigned int, "btc", "I", "l")
inline __attribute__((always_inline))
__buildbitopplus(_btc_u64, unsigned long long, "btc", "J", "q")


/*
   As an aside:
   An atomic version can easily be added using the lock prefix like this:
#define __buildatomicbitopplus(x, y, z, a, b) bool x(y *Base, int Offset) \
{ \
    bool was_set; \
    __asm__ volatile(z "{" b "\t%[Offset],(%[Base]) | (%[Base]),%[Offset]}" \
        : "=@ccc" (was_set), [Base] "+mr" (Base) \
        : [Offset] a "ri" (Offset) \
        : "cc"); \
    return was_set; \
}

// For example:
inline __attribute__((always_inline))
__buildatomicbitopplus(_atomic_btc_u32, unsigned int, "lock btc", "I", "l")

*/

/*
 * int __builtin_popcnt(x)
 * will accept a short x, but will also always add an expand instruction
 * to return an integer. (at least that's what the assembly output says).
 * 
 * This variant does not generate an extraneous MOVZX instruction
 */
inline __attribute__((always_inline))
unsigned short _popcnt_u16(unsigned short inp)
{
    unsigned short ret;
    __asm__ volatile( "popcntw\t {%w[inp], %[ret]|%[ret], %w[inp]}"
        : [ret] "=r" (ret)
        : [inp] "rm" (inp)
        : "cc");
    return ret;
}

// provide some x86_64 instruction intrinsics to a) have a single leading '_' and b) use a _uXX suffix for the operand width
// instead of calling the deviating available form directly
inline unsigned short        _tzcnt_u16(unsigned short x)        { return __tzcnt_u16(x); }
inline unsigned short        _lzcnt_u16(unsigned short x)        { return __lzcnt16(x); }

}

#endif
