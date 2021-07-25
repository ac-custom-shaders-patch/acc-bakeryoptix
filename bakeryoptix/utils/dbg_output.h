#pragma once

#define BOOST_PP_CAT_I(a, b) a ## b
#define BOOST_PP_CAT(a, b) BOOST_PP_CAT_I(a, b)
#define BOOST_PP_VARIADIC_SIZE(...) BOOST_PP_CAT(BOOST_PP_VARIADIC_SIZE_I(__VA_ARGS__, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,),)
#define BOOST_PP_VARIADIC_SIZE_I(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19, e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31, e32, e33, e34, e35, e36, e37, e38, e39, e40, e41, e42, e43, e44, e45, e46, e47, e48, e49, e50, e51, e52, e53, e54, e55, e56, e57, e58, e59, e60, e61, e62, e63, size, ...) size
#define BOOST_PP_OVERLOAD(prefix, ...) BOOST_PP_CAT(prefix, BOOST_PP_VARIADIC_SIZE(__VA_ARGS__))

#define DBG_OUT(V) ", " #V "=" << (V)
#define DBG_1(V) #V "="<< (V)
#define DBG_2(V, V1) #V "="<< (V) << DBG_OUT(V1)
#define DBG_3(V, V1, V2) #V "="<< (V) << DBG_OUT(V1) << DBG_OUT(V2)
#define DBG_4(V, V1, V2, V3) #V "="<< (V) << DBG_OUT(V1) << DBG_OUT(V2) << DBG_OUT(V3)
#define DBG_5(V, V1, V2, V3, V4) #V "="<< (V) << DBG_OUT(V1) << DBG_OUT(V2) << DBG_OUT(V3) << DBG_OUT(V4) 
#define DBG_6(V, V1, V2, V3, V4, V5) #V "="<< (V) << DBG_OUT(V1) << DBG_OUT(V2) << DBG_OUT(V3) << DBG_OUT(V4) << DBG_OUT(V5)  
#define DBG_7(V, V1, V2, V3, V4, V5, V6) #V "="<< (V) << DBG_OUT(V1) << DBG_OUT(V2) << DBG_OUT(V3) << DBG_OUT(V4) << DBG_OUT(V5) << DBG_OUT(V6)  
#define OUTPUT(...) __VA_ARGS__
#define DBG(...) std::cout << "[" << __func__ << ":" << __LINE__ << "] " << OUTPUT(BOOST_PP_OVERLOAD(DBG_, __VA_ARGS__)(__VA_ARGS__)) << "\n";
#define DBG_FN(...) std::cout << "[" << __func__ << ":" << __LINE__ << "] " << OUTPUT(BOOST_PP_OVERLOAD(DBG_, __VA_ARGS__)(__VA_ARGS__)) << "\n";

// #define DEVELOPMENT_CFG
#ifdef DEVELOPMENT_CFG
#define DEBUGTIME __pragma(optimize("", off))
#else
#define DEBUGTIME 
#endif