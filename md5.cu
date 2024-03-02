#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <time.h>

#define N 92 // blocks
#define K 32 // threads

// jsf32 random
typedef uint32_t u4;
typedef struct ranctx { u4 a; u4 b; u4 c; u4 d; } ranctx;

#define rot32(x,k) (((x)<<(k))|((x)>>(32-(k))))
__device__ u4 ranval( ranctx *x ) {
    u4 e = x->a - rot32(x->b, 27);
    x->a = x->b ^ rot32(x->c, 17);
    x->b = x->c + x->d;
    x->c = x->d + e;
    x->d = e + x->a;
    return x->d;
}

__device__ void raninit( ranctx *x, u4 seed ) {
    u4 i;
    x->a = 0xf1ea5eed, x->b = x->c = x->d = seed;
    for (i=0; i<20; ++i) {
        (void)ranval(x);
    }
}

u4 ranval_normal( ranctx *x ) {
    u4 e = x->a - rot32(x->b, 27);
    x->a = x->b ^ rot32(x->c, 17);
    x->b = x->c + x->d;
    x->c = x->d + e;
    x->d = e + x->a;
    return x->d;
}

void raninit_normal( ranctx *x, u4 seed ) {
    u4 i;
    x->a = 0xf1ea5eed, x->b = x->c = x->d = seed;
    for (i=0; i<20; ++i) {
        (void)ranval_normal(x);
    }
}

__device__ void md5_hash_m1(uint32_t* m, uint32_t output[4], uint32_t s[4]) {
    // init
    uint32_t state[4];
    for (int i = 0; i < 4; i++) {
        state[i] = s[i];
    }

    uint32_t A;
    uint32_t B;
    uint32_t C;
    uint32_t D;

    uint32_t AOld;
    uint32_t BOld;
    uint32_t COld;
    uint32_t DOld;

    uint32_t ANew;
    uint32_t BNew;
    uint32_t CNew;
    uint32_t DNew;


#define S(x,n)                                                          \
            ( ( (x) << (n) ) | ( ( (x) & 0xFFFFFFFF) >> ( 32 - (n) ) ) )

#define P(a,b,c,d,k,s,t)                                                \
            do                                                                  \
            {                                                                   \
                (a) += F((b),(c),(d)) + m[(k)] + (t);                     \
                (a) = S((a),(s)) + (b);                                         \
            } while( 0 )

    A = state[0];
    B = state[1];
    C = state[2];
    D = state[3];

    // 16 kroków, jedno F, jeden cykl
#define F(x,y,z) ((z) ^ ((x) & ((y) ^ (z))))
    AOld = A;
    P( A, B, C, D,  0,  7, 0xD76AA478 );
        ANew = (A & 0x71dff7df) | 0x84200000;
        // after applying changes
        if (A != ANew) { // need to compute m1'
            m[0] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
                   - AOld - F(B, C, D) - 0xD76AA478;
            A = ANew;
        }

    DOld = D;
    P( D, A, B, C,  1, 12, 0xE8C7B756 );

        DNew = (D & 0x1c06719) | 0x8c000800 | (A & 0x701f10c0);
        if (D != DNew) { // need to compute m1'
            m[1] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
                   - DOld - F(A, B, C) - 0xE8C7B756;
            D = DNew;
        }

    COld = C;
    P( C, D, A, B,  2, 17, 0x242070DB );

        CNew = (C & 0x1c0e601) | 0xbe1f0966 | (D & 0x00000018);
        if (C != CNew) { // need to compute m1'
            m[2] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
                   - COld - F(D, A, B) - 0x242070DB;
            C = CNew;
        }

    BOld = B;
    P( B, C, D, A,  3, 22, 0xC1BDCEEE );

        BNew = (B & 0x1c0e000) | 0xba040010 | (C & 0x00000601);
        if (B != BNew) { // need to compute m1'
            m[3] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
                   - BOld - F(C, D, A) - 0xC1BDCEEE;
            B = BNew;
        }

    AOld = A;
    P( A, B, C, D,  4,  7, 0xF57C0FAF );

        ANew = (A & 0x3c0e000) | 0x482f0e50;
        if (A != ANew) { // need to compute m1'
            m[4] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
                   - AOld - F(B, C, D) - 0xF57C0FAF;
            A = ANew;
        }

    DOld = D;
    P( D, A, B, C,  5, 12, 0x4787C62A );

        DNew = (D & 0x61cce000) | 0x04220c56;
        if (D != DNew) { // need to compute m1'
            m[5] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
                   - DOld - F(A, B, C) - 0x4787C62A;
            D = DNew;
        }

    COld = C;
    P( C, D, A, B,  6, 17, 0xA8304613 );

        CNew = (C & 0x604c603e) | 0x96011e01 | (D & 0x01808000);
        if (C != CNew) { // need to compute m1'
            m[6] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
                   - COld - F(D, A, B) - 0xA8304613;
            C = CNew;
        }

    BOld = B;
    P( B, C, D, A,  7, 22, 0xFD469501 );

        BNew = (B & 0x604c7c3c) | 0x843283c0 | (C & 0x00000002);
        if (B != BNew) { // need to compute m1'
            m[7] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
                   - BOld - F(C, D, A) - 0xFD469501;
            B = BNew;
        }

    AOld = A;
    P( A, B, C, D,  8,  7, 0x698098D8 );
        ANew = (A & 0x607c6c3c) | 0x9c0101c1 | (B & 0x00001000);
        if (A != ANew) { // need to compute m1'
            m[8] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
                   - AOld - F(B, C, D) - 0x698098D8;
            A = ANew;
        }

    DOld = D;
    P( D, A, B, C,  9, 12, 0x8B44F7AF );
        DNew = (D &  0x78786c3c) | 0x878383c0;
        if (D != DNew) { // need to compute m1'
            m[9] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
                   - DOld - F(A, B, C) - 0x8B44F7AF;
            D = DNew;
        }

    COld = C;
    P( C, D, A, B, 10, 17, 0xFFFF5BB1 );

        CNew = (C & 0x7ff00c3c) | 0x800583c3 | (D & 0x86000);
        if (C != CNew) { // need to compute m1'
            m[10] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
                    - COld - F(D, A, B) - 0xFFFF5BB1;
            C = CNew;
        }

    BOld = B;
    P( B, C, D, A, 11, 22, 0x895CD7BE );

        BNew = (B & 0xf00f7f) | 0x80081080 | (C & 0x7f000000);
        if (B != BNew) { // need to compute m1'
            m[11] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
                    - BOld - F(C, D, A) - 0x895CD7BE;
            B = BNew;
        }

    AOld = A;
    P( A, B, C, D, 12,  7, 0x6B901122 );

        ANew = (A & 0xf01f77) | 0x3f0fe008;
        if (A != ANew) { // need to compute m1'
            m[12] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
                    - AOld - F(B, C, D) - 0x6B901122;
            A = ANew;
        }

    DOld = D;
    P( D, A, B, C, 13, 12, 0xFD987193 );

        DNew = (D & 0xf01f77) | 0x400be088;
        if (D != DNew) { // need to compute m1'
            m[13] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
                    - DOld - F(A, B, C) - 0xFD987193;
            D = DNew;
        }

    COld = C;
    P( C, D, A, B, 14, 17, 0xA679438E );

        CNew = (C & 0xff7ff7) | 0x7d000000;
        if (C != CNew) { // need to compute m1'
            m[14] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
                    - COld - F(D, A, B) - 0xA679438E;
            C = CNew;
        }

    BOld = B;
    P( B, C, D, A, 15, 22, 0x49B40821 );

        BNew = (B & 0x5fffffff) | 0x20000000;
        if (B != BNew) { // need to compute m1'
            m[15] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
                    - BOld - F(C, D, A) - 0x49B40821;
            B = BNew;
        }
#undef F

#define F(x,y,z) ((y) ^ ((z) & ((x) ^ (y))))
    P( A, B, C, D,  1,  5, 0xF61E2562 );
    P( D, A, B, C,  6,  9, 0xC040B340 );
    P( C, D, A, B, 11, 14, 0x265E5A51 );
    P( B, C, D, A,  0, 20, 0xE9B6C7AA );
    P( A, B, C, D,  5,  5, 0xD62F105D );
    P( D, A, B, C, 10,  9, 0x02441453 );
    P( C, D, A, B, 15, 14, 0xD8A1E681 );
    P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
    P( A, B, C, D,  9,  5, 0x21E1CDE6 );
    P( D, A, B, C, 14,  9, 0xC33707D6 );
    P( C, D, A, B,  3, 14, 0xF4D50D87 );
    P( B, C, D, A,  8, 20, 0x455A14ED );
    P( A, B, C, D, 13,  5, 0xA9E3E905 );
    P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
    P( C, D, A, B,  7, 14, 0x676F02D9 );
    P( B, C, D, A, 12, 20, 0x8D2A4C8A );

#undef F

#define F(x,y,z) ((x) ^ (y) ^ (z))
    P( A, B, C, D,  5,  4, 0xFFFA3942 );
    P( D, A, B, C,  8, 11, 0x8771F681 );
    P( C, D, A, B, 11, 16, 0x6D9D6122 );
    P( B, C, D, A, 14, 23, 0xFDE5380C );
    P( A, B, C, D,  1,  4, 0xA4BEEA44 );
    P( D, A, B, C,  4, 11, 0x4BDECFA9 );
    P( C, D, A, B,  7, 16, 0xF6BB4B60 );
    P( B, C, D, A, 10, 23, 0xBEBFBC70 );
    P( A, B, C, D, 13,  4, 0x289B7EC6 );
    P( D, A, B, C,  0, 11, 0xEAA127FA );
    P( C, D, A, B,  3, 16, 0xD4EF3085 );
    P( B, C, D, A,  6, 23, 0x04881D05 );
    P( A, B, C, D,  9,  4, 0xD9D4D039 );
    P( D, A, B, C, 12, 11, 0xE6DB99E5 );
    P( C, D, A, B, 15, 16, 0x1FA27CF8 );
    P( B, C, D, A,  2, 23, 0xC4AC5665 );
#undef F

#define F(x,y,z) ((y) ^ ((x) | ~(z)))
    P( A, B, C, D,  0,  6, 0xF4292244 );
    P( D, A, B, C,  7, 10, 0x432AFF97 );
    P( C, D, A, B, 14, 15, 0xAB9423A7 );
    P( B, C, D, A,  5, 21, 0xFC93A039 );
    P( A, B, C, D, 12,  6, 0x655B59C3 );
    P( D, A, B, C,  3, 10, 0x8F0CCC92 );
    P( C, D, A, B, 10, 15, 0xFFEFF47D );
    P( B, C, D, A,  1, 21, 0x85845DD1 );
    P( A, B, C, D,  8,  6, 0x6FA87E4F );
    P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
    P( C, D, A, B,  6, 15, 0xA3014314 );
    P( B, C, D, A, 13, 21, 0x4E0811A1 );
    P( A, B, C, D,  4,  6, 0xF7537E82 );
    P( D, A, B, C, 11, 10, 0xBD3AF235 );
    P( C, D, A, B,  2, 15, 0x2AD7D2BB );
    P( B, C, D, A,  9, 21, 0xEB86D391 );
#undef F

    state[0] += A;
    state[1] += B;
    state[2] += C;
    state[3] += D;

    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

__device__ void md5_hash(uint32_t* m, uint32_t output[4], uint32_t s[4]) {
    // init
    uint32_t state[4];
    for (int i = 0; i < 4; i++) {
        state[i] = s[i];
    }

    uint32_t A;
    uint32_t B;
    uint32_t C;
    uint32_t D;


#define S(x,n)                                                          \
            ( ( (x) << (n) ) | ( ( (x) & 0xFFFFFFFF) >> ( 32 - (n) ) ) )

#define P(a,b,c,d,k,s,t)                                                \
            do                                                                  \
            {                                                                   \
                (a) += F((b),(c),(d)) + m[(k)] + (t);                     \
                (a) = S((a),(s)) + (b);                                         \
            } while( 0 )

    A = state[0];
    B = state[1];
    C = state[2];
    D = state[3];

    // 16 kroków, jedno F, jeden cykl
#define F(x,y,z) ((z) ^ ((x) & ((y) ^ (z))))
    P( A, B, C, D,  0,  7, 0xD76AA478 );
    P( D, A, B, C,  1, 12, 0xE8C7B756 );
    P( C, D, A, B,  2, 17, 0x242070DB );
    P( B, C, D, A,  3, 22, 0xC1BDCEEE );
    P( A, B, C, D,  4,  7, 0xF57C0FAF );
    P( D, A, B, C,  5, 12, 0x4787C62A );
    P( C, D, A, B,  6, 17, 0xA8304613 );
    P( B, C, D, A,  7, 22, 0xFD469501 );
    P( A, B, C, D,  8,  7, 0x698098D8 );
    P( D, A, B, C,  9, 12, 0x8B44F7AF );
    P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
    P( B, C, D, A, 11, 22, 0x895CD7BE );
    P( A, B, C, D, 12,  7, 0x6B901122 );
    P( D, A, B, C, 13, 12, 0xFD987193 );
    P( C, D, A, B, 14, 17, 0xA679438E );
    P( B, C, D, A, 15, 22, 0x49B40821 );
#undef F

#define F(x,y,z) ((y) ^ ((z) & ((x) ^ (y))))
    P( A, B, C, D,  1,  5, 0xF61E2562 );
    P( D, A, B, C,  6,  9, 0xC040B340 );
    P( C, D, A, B, 11, 14, 0x265E5A51 );
    P( B, C, D, A,  0, 20, 0xE9B6C7AA );
    P( A, B, C, D,  5,  5, 0xD62F105D );
    P( D, A, B, C, 10,  9, 0x02441453 );
    P( C, D, A, B, 15, 14, 0xD8A1E681 );
    P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
    P( A, B, C, D,  9,  5, 0x21E1CDE6 );
    P( D, A, B, C, 14,  9, 0xC33707D6 );
    P( C, D, A, B,  3, 14, 0xF4D50D87 );
    P( B, C, D, A,  8, 20, 0x455A14ED );
    P( A, B, C, D, 13,  5, 0xA9E3E905 );
    P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
    P( C, D, A, B,  7, 14, 0x676F02D9 );
    P( B, C, D, A, 12, 20, 0x8D2A4C8A );

#undef F

#define F(x,y,z) ((x) ^ (y) ^ (z))
    P( A, B, C, D,  5,  4, 0xFFFA3942 );
    P( D, A, B, C,  8, 11, 0x8771F681 );
    P( C, D, A, B, 11, 16, 0x6D9D6122 );
    P( B, C, D, A, 14, 23, 0xFDE5380C );
    P( A, B, C, D,  1,  4, 0xA4BEEA44 );
    P( D, A, B, C,  4, 11, 0x4BDECFA9 );
    P( C, D, A, B,  7, 16, 0xF6BB4B60 );
    P( B, C, D, A, 10, 23, 0xBEBFBC70 );
    P( A, B, C, D, 13,  4, 0x289B7EC6 );
    P( D, A, B, C,  0, 11, 0xEAA127FA );
    P( C, D, A, B,  3, 16, 0xD4EF3085 );
    P( B, C, D, A,  6, 23, 0x04881D05 );
    P( A, B, C, D,  9,  4, 0xD9D4D039 );
    P( D, A, B, C, 12, 11, 0xE6DB99E5 );
    P( C, D, A, B, 15, 16, 0x1FA27CF8 );
    P( B, C, D, A,  2, 23, 0xC4AC5665 );
#undef F

#define F(x,y,z) ((y) ^ ((x) | ~(z)))
    P( A, B, C, D,  0,  6, 0xF4292244 );
    P( D, A, B, C,  7, 10, 0x432AFF97 );
    P( C, D, A, B, 14, 15, 0xAB9423A7 );
    P( B, C, D, A,  5, 21, 0xFC93A039 );
    P( A, B, C, D, 12,  6, 0x655B59C3 );
    P( D, A, B, C,  3, 10, 0x8F0CCC92 );
    P( C, D, A, B, 10, 15, 0xFFEFF47D );
    P( B, C, D, A,  1, 21, 0x85845DD1 );
    P( A, B, C, D,  8,  6, 0x6FA87E4F );
    P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
    P( C, D, A, B,  6, 15, 0xA3014314 );
    P( B, C, D, A, 13, 21, 0x4E0811A1 );
    P( A, B, C, D,  4,  6, 0xF7537E82 );
    P( D, A, B, C, 11, 10, 0xBD3AF235 );
    P( C, D, A, B,  2, 15, 0x2AD7D2BB );
    P( B, C, D, A,  9, 21, 0xEB86D391 );
#undef F

    state[0] += A;
    state[1] += B;
    state[2] += C;
    state[3] += D;

    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

__global__ void attack(uint32_t seed, int* success) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
   // success[idx] = 0;
    // m0 hashed (initial state for m1)
//    uint32_t s1[] = {0x52589324, 0x3093d7ca, 0x2a06dc54, 0x20c5be06};
    // m0' hashed (initial state for m1')
//    uint32_t s2[] = {0xd2589324, 0xb293d7ca, 0xac06dc54, 0xa2c5be06};
	uint32_t s1[] = {0xfec19f38, 0xf430a8ea, 0x8ab480e4, 0x58892490};
    uint32_t s2[] = {0x7ec19f38, 0x7630a8ea, 0xcb480e4, 0xda892490};
    uint32_t m1[16];
    uint32_t m1_added[16];
    uint32_t digest1[4];
    uint32_t digest2[4];
    int equal = 0;
    // initialize random generator
    struct ranctx state;
    raninit(&state, seed + idx);
    // one thread performs 2^20 experiments
    for (int j = 0; j < 1048576 ; j++) {
        // get random message m1
        for (int i = 0; i < 16; i++) {
            m1[i] = ranval(&state);
       //     d_messages[idx + i] = m1[i];
        }

        // modify and check if this works
        md5_hash_m1(m1, digest1, s1);

        // construct m1'
        for (int i = 0; i < 16; i++) {
            m1_added[i] = m1[i];
        }
        m1_added[4] += 0x80000000;
        m1_added[11] -= 0x00008000;
        m1_added[14] += 0x80000000;

        md5_hash(m1_added, digest2, s2);
       // printf("\n%x -------------------------------\n", digest2[0]);

        // check if hashes equal
        equal = 1;
        for (int i = 0; i < 4; i++) {
            if (digest1[i] != digest2[i]) {
                equal = 0;
            }
        }
        if (equal == 1) { // collision found
            success[idx] = 1;
            break;
        }
    }
}

void md5_hash_normal(uint32_t* m, uint32_t output[4], uint32_t* s) {
    uint32_t state[4];
    for (int i = 0; i < 4; i++) {
        state[i] = s[i];
    }

    uint32_t A;
    uint32_t B;
    uint32_t C;
    uint32_t D;

#define S(x,n)                                                          \
            ( ( (x) << (n) ) | ( ( (x) & 0xFFFFFFFF) >> ( 32 - (n) ) ) )

#define P(a,b,c,d,k,s,t)                                                \
            do                                                                  \
            {                                                                   \
                (a) += F((b),(c),(d)) + m[(k)] + (t);                     \
                (a) = S((a),(s)) + (b);                                         \
            } while( 0 )

    A = state[0];
    B = state[1];
    C = state[2];
    D = state[3];

    // 16 kroków, jedno F, jeden cykl
#define F(x,y,z) ((z) ^ ((x) & ((y) ^ (z))))
    P( A, B, C, D,  0,  7, 0xD76AA478 );
    P( D, A, B, C,  1, 12, 0xE8C7B756 );
    P( C, D, A, B,  2, 17, 0x242070DB );
    P( B, C, D, A,  3, 22, 0xC1BDCEEE );
    P( A, B, C, D,  4,  7, 0xF57C0FAF );
    P( D, A, B, C,  5, 12, 0x4787C62A );
    P( C, D, A, B,  6, 17, 0xA8304613 );
    P( B, C, D, A,  7, 22, 0xFD469501 );
    P( A, B, C, D,  8,  7, 0x698098D8 );
    P( D, A, B, C,  9, 12, 0x8B44F7AF );
    P( C, D, A, B, 10, 17, 0xFFFF5BB1 );
    P( B, C, D, A, 11, 22, 0x895CD7BE );
    P( A, B, C, D, 12,  7, 0x6B901122 );
    P( D, A, B, C, 13, 12, 0xFD987193 );
    P( C, D, A, B, 14, 17, 0xA679438E );
    P( B, C, D, A, 15, 22, 0x49B40821 );
#undef F

#define F(x,y,z) ((y) ^ ((z) & ((x) ^ (y))))
    P( A, B, C, D,  1,  5, 0xF61E2562 );
    P( D, A, B, C,  6,  9, 0xC040B340 );
    P( C, D, A, B, 11, 14, 0x265E5A51 );
    P( B, C, D, A,  0, 20, 0xE9B6C7AA );
    P( A, B, C, D,  5,  5, 0xD62F105D );
    P( D, A, B, C, 10,  9, 0x02441453 );
    P( C, D, A, B, 15, 14, 0xD8A1E681 );
    P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
    P( A, B, C, D,  9,  5, 0x21E1CDE6 );
    P( D, A, B, C, 14,  9, 0xC33707D6 );
    P( C, D, A, B,  3, 14, 0xF4D50D87 );
    P( B, C, D, A,  8, 20, 0x455A14ED );
    P( A, B, C, D, 13,  5, 0xA9E3E905 );
    P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
    P( C, D, A, B,  7, 14, 0x676F02D9 );
    P( B, C, D, A, 12, 20, 0x8D2A4C8A );

#undef F

#define F(x,y,z) ((x) ^ (y) ^ (z))
    P( A, B, C, D,  5,  4, 0xFFFA3942 );
    P( D, A, B, C,  8, 11, 0x8771F681 );
    P( C, D, A, B, 11, 16, 0x6D9D6122 );
    P( B, C, D, A, 14, 23, 0xFDE5380C );
    P( A, B, C, D,  1,  4, 0xA4BEEA44 );
    P( D, A, B, C,  4, 11, 0x4BDECFA9 );
    P( C, D, A, B,  7, 16, 0xF6BB4B60 );
    P( B, C, D, A, 10, 23, 0xBEBFBC70 );
    P( A, B, C, D, 13,  4, 0x289B7EC6 );
    P( D, A, B, C,  0, 11, 0xEAA127FA );
    P( C, D, A, B,  3, 16, 0xD4EF3085 );
    P( B, C, D, A,  6, 23, 0x04881D05 );
    P( A, B, C, D,  9,  4, 0xD9D4D039 );
    P( D, A, B, C, 12, 11, 0xE6DB99E5 );
    P( C, D, A, B, 15, 16, 0x1FA27CF8 );
    P( B, C, D, A,  2, 23, 0xC4AC5665 );
#undef F

#define F(x,y,z) ((y) ^ ((x) | ~(z)))
    P( A, B, C, D,  0,  6, 0xF4292244 );
    P( D, A, B, C,  7, 10, 0x432AFF97 );
    P( C, D, A, B, 14, 15, 0xAB9423A7 );
    P( B, C, D, A,  5, 21, 0xFC93A039 );
    P( A, B, C, D, 12,  6, 0x655B59C3 );
    P( D, A, B, C,  3, 10, 0x8F0CCC92 );
    P( C, D, A, B, 10, 15, 0xFFEFF47D );
    P( B, C, D, A,  1, 21, 0x85845DD1 );
    P( A, B, C, D,  8,  6, 0x6FA87E4F );
    P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
    P( C, D, A, B,  6, 15, 0xA3014314 );
    P( B, C, D, A, 13, 21, 0x4E0811A1 );
    P( A, B, C, D,  4,  6, 0xF7537E82 );
    P( D, A, B, C, 11, 10, 0xBD3AF235 );
    P( C, D, A, B,  2, 15, 0x2AD7D2BB );
    P( B, C, D, A,  9, 21, 0xEB86D391 );
#undef F

    state[0] += A;
    state[1] += B;
    state[2] += C;
    state[3] += D;

    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

void md5_hash_m1_normal(uint32_t* m, uint32_t output[4], uint32_t s[4]) {
    // init
    uint32_t state[4];
    for (int i = 0; i < 4; i++) {
        state[i] = s[i];
    }

    uint32_t A;
    uint32_t B;
    uint32_t C;
    uint32_t D;

    uint32_t AOld;
    uint32_t BOld;
    uint32_t COld;
    uint32_t DOld;

    uint32_t ANew;
    uint32_t BNew;
    uint32_t CNew;
    uint32_t DNew;


#define S(x,n)                                                          \
            ( ( (x) << (n) ) | ( ( (x) & 0xFFFFFFFF) >> ( 32 - (n) ) ) )

#define P(a,b,c,d,k,s,t)                                                \
            do                                                                  \
            {                                                                   \
                (a) += F((b),(c),(d)) + m[(k)] + (t);                     \
                (a) = S((a),(s)) + (b);                                         \
            } while( 0 )

    A = state[0];
    B = state[1];
    C = state[2];
    D = state[3];

    // 16 kroków, jedno F, jeden cykl
#define F(x,y,z) ((z) ^ ((x) & ((y) ^ (z))))
    AOld = A;
    P( A, B, C, D,  0,  7, 0xD76AA478 );
    ANew = (A & 0x71dff7df) | 0x84200000;
    // after applying changes
    if (A != ANew) { // need to compute m1'
        m[0] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
               - AOld - F(B, C, D) - 0xD76AA478;
        A = ANew;
    }

    DOld = D;
    P( D, A, B, C,  1, 12, 0xE8C7B756 );

    DNew = (D & 0x1c06719) | 0x8c000800 | (A & 0x701f10c0);
    if (D != DNew) { // need to compute m1'
        m[1] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
               - DOld - F(A, B, C) - 0xE8C7B756;
        D = DNew;
    }

    COld = C;
    P( C, D, A, B,  2, 17, 0x242070DB );

    CNew = (C & 0x1c0e601) | 0xbe1f0966 | (D & 0x00000018);
    if (C != CNew) { // need to compute m1'
        m[2] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
               - COld - F(D, A, B) - 0x242070DB;
        C = CNew;
    }

    BOld = B;
    P( B, C, D, A,  3, 22, 0xC1BDCEEE );

    BNew = (B & 0x1c0e000) | 0xba040010 | (C & 0x00000601);
    if (B != BNew) { // need to compute m1'
        m[3] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
               - BOld - F(C, D, A) - 0xC1BDCEEE;
        B = BNew;
    }

    AOld = A;
    P( A, B, C, D,  4,  7, 0xF57C0FAF );

    ANew = (A & 0x3c0e000) | 0x482f0e50;
    if (A != ANew) { // need to compute m1'
        m[4] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
               - AOld - F(B, C, D) - 0xF57C0FAF;
        A = ANew;
    }

    DOld = D;
    P( D, A, B, C,  5, 12, 0x4787C62A );

    DNew = (D & 0x61cce000) | 0x04220c56;
    if (D != DNew) { // need to compute m1'
        m[5] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
               - DOld - F(A, B, C) - 0x4787C62A;
        D = DNew;
    }

    COld = C;
    P( C, D, A, B,  6, 17, 0xA8304613 );

    CNew = (C & 0x604c603e) | 0x96011e01 | (D & 0x01808000);
    if (C != CNew) { // need to compute m1'
        m[6] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
               - COld - F(D, A, B) - 0xA8304613;
        C = CNew;
    }

    BOld = B;
    P( B, C, D, A,  7, 22, 0xFD469501 );

    BNew = (B & 0x604c7c3c) | 0x843283c0 | (C & 0x00000002);
    if (B != BNew) { // need to compute m1'
        m[7] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
               - BOld - F(C, D, A) - 0xFD469501;
        B = BNew;
    }

    AOld = A;
    P( A, B, C, D,  8,  7, 0x698098D8 );
    ANew = (A & 0x607c6c3c) | 0x9c0101c1 | (B & 0x00001000);
    if (A != ANew) { // need to compute m1'
        m[8] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
               - AOld - F(B, C, D) - 0x698098D8;
        A = ANew;
    }

    DOld = D;
    P( D, A, B, C,  9, 12, 0x8B44F7AF );
    DNew = (D &  0x78786c3c) | 0x878383c0;
    if (D != DNew) { // need to compute m1'
        m[9] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
               - DOld - F(A, B, C) - 0x8B44F7AF;
        D = DNew;
    }

    COld = C;
    P( C, D, A, B, 10, 17, 0xFFFF5BB1 );

    CNew = (C & 0x7ff00c3c) | 0x800583c3 | (D & 0x86000);
    if (C != CNew) { // need to compute m1'
        m[10] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
                - COld - F(D, A, B) - 0xFFFF5BB1;
        C = CNew;
    }

    BOld = B;
    P( B, C, D, A, 11, 22, 0x895CD7BE );

    BNew = (B & 0xf00f7f) | 0x80081080 | (C & 0x7f000000);
    if (B != BNew) { // need to compute m1'
        m[11] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
                - BOld - F(C, D, A) - 0x895CD7BE;
        B = BNew;
    }

    AOld = A;
    P( A, B, C, D, 12,  7, 0x6B901122 );

    ANew = (A & 0xf01f77) | 0x3f0fe008;
    if (A != ANew) { // need to compute m1'
        m[12] = ((((ANew - B)) >> (7)) | ((((ANew - B) ) & 0xFFFFFFFF) << (32 - (7))))
                - AOld - F(B, C, D) - 0x6B901122;
        A = ANew;
    }

    DOld = D;
    P( D, A, B, C, 13, 12, 0xFD987193 );

    DNew = (D & 0xf01f77) | 0x400be088;
    if (D != DNew) { // need to compute m1'
        m[13] = ((((DNew - A)) >> (12)) | ((((DNew - A) ) & 0xFFFFFFFF) << (32 - (12))))
                - DOld - F(A, B, C) - 0xFD987193;
        D = DNew;
    }

    COld = C;
    P( C, D, A, B, 14, 17, 0xA679438E );

    CNew = (C & 0xff7ff7) | 0x7d000000;
    if (C != CNew) { // need to compute m1'
        m[14] = ((((CNew - D)) >> (17)) | ((((CNew - D) ) & 0xFFFFFFFF) << (32 - (17))))
                - COld - F(D, A, B) - 0xA679438E;
        C = CNew;
    }

    BOld = B;
    P( B, C, D, A, 15, 22, 0x49B40821 );

    BNew = (B & 0x5fffffff) | 0x20000000;
    if (B != BNew) { // need to compute m1'
        m[15] = ((((BNew - C)) >> (22)) | ((((BNew - C) ) & 0xFFFFFFFF) << (32 - (22))))
                - BOld - F(C, D, A) - 0x49B40821;
        B = BNew;
    }
#undef F

#define F(x,y,z) ((y) ^ ((z) & ((x) ^ (y))))
    P( A, B, C, D,  1,  5, 0xF61E2562 );
    P( D, A, B, C,  6,  9, 0xC040B340 );
    P( C, D, A, B, 11, 14, 0x265E5A51 );
    P( B, C, D, A,  0, 20, 0xE9B6C7AA );
    P( A, B, C, D,  5,  5, 0xD62F105D );
    P( D, A, B, C, 10,  9, 0x02441453 );
    P( C, D, A, B, 15, 14, 0xD8A1E681 );
    P( B, C, D, A,  4, 20, 0xE7D3FBC8 );
    P( A, B, C, D,  9,  5, 0x21E1CDE6 );
    P( D, A, B, C, 14,  9, 0xC33707D6 );
    P( C, D, A, B,  3, 14, 0xF4D50D87 );
    P( B, C, D, A,  8, 20, 0x455A14ED );
    P( A, B, C, D, 13,  5, 0xA9E3E905 );
    P( D, A, B, C,  2,  9, 0xFCEFA3F8 );
    P( C, D, A, B,  7, 14, 0x676F02D9 );
    P( B, C, D, A, 12, 20, 0x8D2A4C8A );

#undef F

#define F(x,y,z) ((x) ^ (y) ^ (z))
    P( A, B, C, D,  5,  4, 0xFFFA3942 );
    P( D, A, B, C,  8, 11, 0x8771F681 );
    P( C, D, A, B, 11, 16, 0x6D9D6122 );
    P( B, C, D, A, 14, 23, 0xFDE5380C );
    P( A, B, C, D,  1,  4, 0xA4BEEA44 );
    P( D, A, B, C,  4, 11, 0x4BDECFA9 );
    P( C, D, A, B,  7, 16, 0xF6BB4B60 );
    P( B, C, D, A, 10, 23, 0xBEBFBC70 );
    P( A, B, C, D, 13,  4, 0x289B7EC6 );
    P( D, A, B, C,  0, 11, 0xEAA127FA );
    P( C, D, A, B,  3, 16, 0xD4EF3085 );
    P( B, C, D, A,  6, 23, 0x04881D05 );
    P( A, B, C, D,  9,  4, 0xD9D4D039 );
    P( D, A, B, C, 12, 11, 0xE6DB99E5 );
    P( C, D, A, B, 15, 16, 0x1FA27CF8 );
    P( B, C, D, A,  2, 23, 0xC4AC5665 );
#undef F

#define F(x,y,z) ((y) ^ ((x) | ~(z)))
    P( A, B, C, D,  0,  6, 0xF4292244 );
    P( D, A, B, C,  7, 10, 0x432AFF97 );
    P( C, D, A, B, 14, 15, 0xAB9423A7 );
    P( B, C, D, A,  5, 21, 0xFC93A039 );
    P( A, B, C, D, 12,  6, 0x655B59C3 );
    P( D, A, B, C,  3, 10, 0x8F0CCC92 );
    P( C, D, A, B, 10, 15, 0xFFEFF47D );
    P( B, C, D, A,  1, 21, 0x85845DD1 );
    P( A, B, C, D,  8,  6, 0x6FA87E4F );
    P( D, A, B, C, 15, 10, 0xFE2CE6E0 );
    P( C, D, A, B,  6, 15, 0xA3014314 );
    P( B, C, D, A, 13, 21, 0x4E0811A1 );
    P( A, B, C, D,  4,  6, 0xF7537E82 );
    P( D, A, B, C, 11, 10, 0xBD3AF235 );
    P( C, D, A, B,  2, 15, 0x2AD7D2BB );
    P( B, C, D, A,  9, 21, 0xEB86D391 );
#undef F

    state[0] += A;
    state[1] += B;
    state[2] += C;
    state[3] += D;

    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

int main() {
    int size_of_message = 16;
    // N * K + 16 * i
    uint32_t mess1[16];
    uint32_t digest[4];
   // uint32_t* messages = (uint32_t*)malloc(sizeof(uint32_t) * N * K * size_of_message);
   // uint32_t* d_messages;
   // cudaMalloc(&d_messages, sizeof(uint32_t) * N * K * size_of_message);
    int* success = (int*)malloc(sizeof(int) * N * K);
    int* d_success;
    cudaMalloc(&d_success, sizeof(int) * N * K);

    clock_t start, end;
    double elapsed;

    uint32_t lastSeed = 0xabcdef;
    for (int i = 0; i < 1024 * 10; i++) {
        printf("Iteration: %d\n", i);
        start = clock();
        // run kernel
        attack<<<N, K>>>(lastSeed, d_success);
       // cudaMemcpy(messages, d_messages, sizeof(uint32_t) * N * K * size_of_message, cudaMemcpyDeviceToHost);
        cudaMemcpy(success, d_success, sizeof(int) * N * K, cudaMemcpyDeviceToHost);
        // check if collision was found
        for (int j = 0; j < N * K; j++) {
            if (success[j] == 1) { // (j is idx of thread)
                // collision found
                printf("COLLISION FOUND\n");
                // perform experiment but on host with last seed
//                uint32_t s1[] = {0x52589324, 0x3093d7ca, 0x2a06dc54, 0x20c5be06};
//                // m0' hashed (initial state for m1')
//                uint32_t s2[] = {0xd2589324, 0xb293d7ca, 0xac06dc54, 0xa2c5be06};
                uint32_t s1[] = {0xfec19f38, 0xf430a8ea, 0x8ab480e4, 0x58892490};
                uint32_t s2[] = {0x7ec19f38, 0x7630a8ea, 0xcb480e4, 0xda892490};

                uint32_t m1[16];
                uint32_t m1_added[16];
                uint32_t digest1[4];
                uint32_t digest2[4];
                int equal = 0;
                // initialize random generator
                struct ranctx state;
                raninit_normal(&state, lastSeed + j);
                // one thread performs 2^20 experiments
                for (int n = 0; n < 1048576 ; n++) {
                    // get random message m1
                    for (int l = 0; l < 16; l++) {
                        m1[l] = ranval_normal(&state);
                    }

                    // modify and check if this works
                    md5_hash_m1_normal(m1, digest1, s1);

                    // construct m1'
                    for (int l = 0; l < 16; l++) {
                        m1_added[l] = m1[l];
                    }
                    m1_added[4] += 0x80000000;
                    m1_added[11] -= 0x00008000;
                    m1_added[14] += 0x80000000;

                    md5_hash_normal(m1_added, digest2, s2);

                    // check if hashes equal
                    equal = 1;
                    for (int i = 0; i < 4; i++) {
                        if (digest1[i] != digest2[i]) {
                            equal = 0;
                        }
                    }
                    if (equal == 1) { // collision found
                        printf("m1 \n");
                        for (int k = 0; k < 16; k++) {
                            printf("%x ", m1[k]);
                        }
                        printf("\n");
                        printf("m1' \n");
                        for (int k = 0; k < 16; k++) {
                            printf("%x ", m1_added[k]);
                        }
                        printf("\nhash \n");
                        for (int k = 0; k < 4; k++) {
                            printf("%x ", digest1[k]);
                        }
                        printf("\n");
                        return;
                    }
                }
            }

        }
        end = clock();
        elapsed = double(end - start) / CLOCKS_PER_SEC;
        printf("Time: %f, Iteration: %d, Last Seed: %x \n", elapsed, i, lastSeed);
        lastSeed += N * K;
    }

//    cudaFree(d_messages);
    cudaFree(d_success);
//    free(messages);
    free(success);
}

/*
 * collision found - iteration 4016, seed  1602a6f + 92*32
 * m1
 * dafde4f7 4bac395a 6bd85fe6 e4685ada 1581f7f8 1d119be8 572d1d2a d1616bd4 9d3f0fb9 cb9bbb79 7d3ffaf6 cf4e1499 c7755cd9 4df3ff84 6d2a53b9 2c72018d
 * m1'
 * dafde4f7 4bac395a 6bd85fe6 e4685ada 9581f7f8 1d119be8 572d1d2a d1616bd4 9d3f0fb9 cb9bbb79 7d3ffaf6 cf4d9499 c7755cd9 4df3ff84 ed2a53b9 2c72018d
 * hash
 * 246da77 60ce90a4 148fc85f fd34275
 */
