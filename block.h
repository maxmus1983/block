#pragma once

//This file is copyrighted by christoph keller and is used under the BSD license

#include <smmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <utility>
#include <algorithm>
#include <cmath>
#include <limits>
#include <typeinfo>
#include <malloc.h>
#include <assert.h>

#define INCLUDE_AVX
#define INCLUDE_AVX512

#include "SB2_NoPCH/ssehelpers.h"

#define MAKE_ALIGNED_OPERATOR_NEW \
	void* operator new(size_t numB) { return _aligned_malloc(numB, 64); } \
	void  operator delete(void* del) { return _aligned_free(del); } \
	void* operator new[](size_t numB) { return _aligned_malloc(numB, 64); } \
	void  operator delete[](void* del) { return _aligned_free(del); }

namespace RBEN {

	template <class From, class To> inline To memcnv(From fr) { static_assert(sizeof(From)==sizeof(To), "The size needs to be the same"); return *((To*)&fr); }
	inline unsigned memcnvFI(float from) { return memcnv<float, unsigned>(from); }
	inline unsigned long long memcnvDI(double from) { return memcnv<double, unsigned long long>(from); }
	inline float memcnvIF(unsigned from) { return memcnv<unsigned, float>(from); }
	inline double memcnvID(unsigned long long from) { return memcnv<unsigned long long, double>(from); }
	inline float memcnvZero() { return memcnvIF(0); }
	inline float memcnvFull() { return memcnvIF(0xFFFF'FFFF); }
	inline double memcnvFullD() { return memcnvID(0xFFFF'FFFF'FFFF'FFFF); }
	inline bool isAligned(int alignment, float* val) { return (((long long)val)&(alignment - 1)) == 0; }
	inline __m128i ss_set1_si(unsigned val) { __m128i ret; ((unsigned*)&ret)[0] = val; return _mm_shuffle_epi32(ret, _MM_SHUFFLE(0, 0, 0, 0)); }

	//the main class
	template <typename Base> class RBEMom;

	//used to create some properties for RBEMoms
	template <typename T> class rb_traits;

	//the types of accelerations available
	enum AcType : int
	{
		AC_BASE = 0,
		AC_NONE = 1,
		AC_SSE = 4,
		AC_AVX = 8,
		AC_AVX512 = 16
	};

	template <AcType T, typename DType = float> class Packet;
	enum PacketInitOptions
	{
		PIO_SETZERO,
		PIO_SETONES,
		PIO_NONE
	};

	template <typename T>
	class Packet<AC_NONE, T>
	{
	public:
		enum Options {
			len = 1,
			Mfull = 1 //Mfull is the maximum bitmask that one can get from the _mmXXX_mask_pX instructions
		};
		T val;
		inline operator T() const { return val; }
		inline Packet(T* adr) { load(adr); }
		inline Packet(T  iVal) { val = iVal; }
		inline Packet() {}
		inline Packet(PacketInitOptions TVal) { if (TVal == PIO_SETZERO) val = 0; }
		inline void load(T* adr) { val = *adr; }
		inline void load1(const T* adr) { val = *adr; }
		inline void load1(T adr) { val = adr; }
		inline void store(T* adr) { *adr = val; }
		inline int mask() const { return val > 0 ? 1 : 0; }
		inline T &operator[](int i) { assert(i == 0); return val; }
		inline const T &operator[](int i) const { assert(i == 0); return val; }
	};

	template <>
	class
#ifdef ALIGNMENT_OPTIONS	
		ALIGNMENT_OPTIONS
#endif	
		Packet<AC_SSE, float>
	{
	public:
		__m128 val;
		enum Options {
			len = 4,
			Mfull = 15
		};
		inline operator __m128() const { return val; }
		inline Packet(__m128  iVal) { val = iVal; }
		inline Packet(float* adr) { load(adr); }
		inline Packet(float val) { load1(val); }
		inline Packet() {}
		inline Packet(PacketInitOptions T) {
			if (T == PIO_SETZERO)
				val = _mm_setzero_ps();
			if (T == PIO_SETONES) {
				__m128 tmp = _mm_setzero_ps();
				val = _mm_cmpeq_ps(tmp, tmp);
			}
		}

		template <int sTo>
		inline Packet<AC_SSE, float> allTo() { return _mm_shuffle_ps(val, val, _MM_SHUFFLE(sTo, sTo, sTo, sTo)); }
		inline void load(float* adr) { val = _mm_load_ps(adr); }
		inline void load1(const float* adr) { val = _mm_load1_ps(adr); }
		inline void load1(float adr) { val = _mm_set1_ps(adr); }
		inline void store(float* adr) { _mm_store_ps(adr, val); }
		inline int mask() const { return _mm_movemask_ps(val); }
		inline float &operator[](int i) const { return ((float*)this)[i]; }
	};

	template <>
	class
#ifdef ALIGNMENT_OPTIONS	
		ALIGNMENT_OPTIONS
#endif	
		Packet<AC_SSE, unsigned>
	{
	public:
		__m128i val;
		enum Options {
			len = 4,
			Mfull = 15
		};
		inline operator __m128i() const { return val; }
		inline Packet(const __m128i &iVal) { val = iVal; }
		inline Packet(unsigned* adr) { load(adr); }
		inline Packet(unsigned adr) { load1(adr); }
		inline Packet() {}
		inline Packet(PacketInitOptions T) {
			if (T == PIO_SETZERO) val = _mm_setzero_si128();
			if (T == PIO_SETONES) {
				__m128i tmp = _mm_setzero_si128();
				val = _mm_cmpeq_epi32(tmp, tmp);
			}
		}

		template <int sslN> inline Packet<AC_SSE, unsigned> slli() { return _mm_slli_epi32(val, sslN); }
		template <int ssrN> inline Packet<AC_SSE, unsigned> srli() { return _mm_srli_si128(val, ssrN); }
		inline void load(unsigned* adr) { val = _mm_load_si128((__m128i*)adr); }
		inline void load1(const unsigned* adr) { load1(*adr); }
		inline void load1(unsigned adr) { val = _mm_set1_epi32(adr); }
		inline void store(unsigned* adr) { _mm_store_si128((__m128i*)adr, val); }
		inline unsigned &operator[](int i) const { return ((unsigned*)this)[i]; }
		inline unsigned &get(int i) { return ((unsigned*)this)[i]; }
	};

	template <>
	class
#ifdef ALIGNMENT_OPTIONS	
		ALIGNMENT_OPTIONS
#endif	
		Packet<AC_SSE, double>
	{
	public:
		__m128d val;
		enum Options {
			len = 2,
			Mfull = 3
		};
		inline operator __m128d() const { return val; }
		inline Packet(const __m128d& iVal) { val = iVal; }
		inline Packet(double* adr) { load(adr); }
		inline Packet(double adr) { load1(adr); }
		inline Packet() {}
		inline Packet(PacketInitOptions T) {
			if (T == PIO_SETZERO) val = _mm_setzero_pd();
			if (T == PIO_SETONES) {
				__m128d tmp = _mm_setzero_pd();
				val = _mm_cmpeq_pd(tmp, tmp);
			}
		}
		inline int mask() const { return _mm_movemask_pd(val); }
		inline void load(double* adr) { val = _mm_load_pd(adr); }
		inline void load1(const double* adr) { load1(*adr); }
		inline void load1(double adr) { val = _mm_set1_pd(adr); }
		inline void store(double* adr) { _mm_store_pd(adr, val); }
		inline double& operator[](int i) const { return ((double*)this)[i]; }
		inline double& get(int i) { return ((double*)this)[i]; }
	};

#ifdef INCLUDE_AVX
#pragma warning(push)
#pragma warning(disable:4752) // found Intel(R) Advanced Vector Extensions
	template <>
	class Packet<AC_AVX, float>
	{
	public:
		__m256 val;
		enum Options {
			len = 8,
			Mfull = 255
		};
		inline operator __m256() const { return val; }
		inline Packet(__m256  iVal) { val = iVal; }
		inline Packet(const float* adr) { load(adr); }
		inline Packet(float val) { load1(val); }
		inline Packet() {}
		inline Packet(PacketInitOptions T) {
			if (T == PIO_SETZERO) val = _mm256_setzero_ps();
			if (T == PIO_SETONES) {
				__m256 tmp = _mm256_setzero_ps();
				val = _mm256_cmp_ps(tmp, tmp, 0);
			}
		}
		inline int mask() const { return _mm256_movemask_ps(val); }
		inline void load(const float* adr) { val = _mm256_load_ps(adr); }
		inline void load1(const float* adr) { val = _mm256_set1_ps(*adr); }
		inline void load1(float adr) { val = _mm256_set1_ps(adr); }
		inline void store(float* adr) { _mm256_store_ps(adr, val); }
		inline float &operator[](int i) const { return ss_getf(val, i); }
	};

	template <>
	class Packet<AC_AVX, double>
	{
	public:
		__m256d val;
		enum Options {
			len = 4,
			Mfull = 15
		};
		inline operator __m256d() const { return val; }
		inline Packet(__m256d  iVal) { val = iVal; }
		inline Packet(const double* adr) { load(adr); }
		inline Packet(double val) { load1(val); }
		inline Packet() {}
		inline Packet(PacketInitOptions T) {
			if (T == PIO_SETZERO) val = _mm256_setzero_pd();
			if (T == PIO_SETONES) {
				__m256d tmp = _mm256_setzero_pd();
				val = _mm256_cmp_pd(tmp, tmp, 0);
			}
		}
		inline int mask() const { return _mm256_movemask_pd(val); }
		inline void load(const double* adr) { val = _mm256_load_pd(adr); }
		inline void load1(const double* adr) { val = _mm256_set1_pd(*adr); }
		inline void load1(double adr) { val = _mm256_set1_pd(adr); }
		inline void store(double* adr) { _mm256_store_pd(adr, val); }
		inline double& operator[](int i) { return val.m256d_f64[i]; }
		inline const double& operator[](int i) const { return val.m256d_f64[i]; }
	};
#pragma warning(pop)
#endif

#ifdef INCLUDE_AVX512
#pragma warning(push)
#pragma warning(disable:4752) // found Intel(R) Advanced Vector Extensions

	class AVX512MASK_16
	{
	public:
		__mmask16 data;
		AVX512MASK_16() { data = 0; }
		AVX512MASK_16(const __mmask16& val) { data = val; }
		int mask() const { return (int)data; }

		AVX512MASK_16 operator||(const AVX512MASK_16& val) const { return _mm512_kor (data, val.data); }
		AVX512MASK_16 operator&&(const AVX512MASK_16& val) const { return _mm512_kand(data, val.data); }
		AVX512MASK_16 operator|(const AVX512MASK_16& val) const { return _mm512_kor(data, val.data); }
		AVX512MASK_16 operator&(const AVX512MASK_16& val) const { return  _mm512_kand(data, val.data); }
	};

	template <>
	class Packet<AC_AVX512, float>
	{
	public:
		__m512 val;
		enum Options {
			len = 16,
			Mfull = 0xFF
		};
		inline operator __m512() const { return val; }
		inline Packet(__m512  iVal) { val = iVal; }
		inline Packet(const float* adr) { load(adr); }
		inline Packet(float val) { load1(val); }
		inline Packet() {}
		inline Packet(PacketInitOptions T) {
			if (T == PIO_SETZERO) val = _mm512_setzero_ps();
			else if (T == PIO_SETONES) {
				const unsigned long dwFull = 0xFFFFFFFF;
				load((float*)&dwFull);
			}
		}
		inline int mask() const { return _mm512_cmplt_ps_mask(val, _mm512_setzero_ps()); }
		inline void load(const float* adr) { val = _mm512_load_ps(adr); }
		inline void load1(const float* adr) { val = _mm512_set1_ps(*adr); }
		inline void load1(float adr) { val = _mm512_set1_ps(adr); }
		inline void store(float* adr) { _mm512_store_ps(adr, val); }
		inline float &operator[](int i) const { return ss_getf(val, i); }
	};
#pragma warning(pop)
#endif

	template <int Tdim, int Tlen, typename FBaseType = float>
	class FStorage {
	public:
		typedef FBaseType SType;
		FBaseType data[Tdim*Tlen];
	};

	template <int Tdim, int Tlen, typename PBaseType = float>
	class PStorage {
	public:
		typedef PBaseType SType;

		PStorage() {}
		PStorage(PBaseType* nDat) { data = nDat; }
		PBaseType *data;
	};

	template <int Tdim, int Tlen, typename BaseType> class PStore;

	template <int Tdim, int Tlen, typename CRetType, class CStorage = FStorage<Tdim, Tlen, float> >
	class
		StoreBase : public CStorage
	{
	public:
		enum Options {
			TmaxPacket = AC_SSE,
			Tsize = Tdim*Tlen
		};
		typedef Packet<(AcType)TmaxPacket> maxPacket;

		template <AcType T> inline Packet<T, CRetType>&
			get(int row, int col) { return *(((Packet<T, CRetType>*)(CStorage::data + row*Tlen)) + col); }
		template <AcType T> inline Packet<T, CRetType>&
			get(int idx) { return *(((Packet<T, CRetType>*)CStorage::data) + idx); }
		inline CRetType &operator[](int i) { return CStorage::data[i]; }

		template <class T>
		inline void setFromAr(T &arg) {
			maxPacket tmp;
			for (int i = 0; i < Tdim; i++) {
				tmp.load1(arg[i]);
				for (int j = 0; j < Tlen / maxPacket::len; j++) {
					get<(AcType)TmaxPacket>(i, j) = tmp;
				}
			}
		}
		inline int mask() {
			int ret = 0;
			for (int i = 0; i < Tsize; i += maxPacket::len) {
				maxPacket mP(CStorage::data + i);
				ret = ret << maxPacket::len;
				ret |= mP.mask();
			}
			return ret;
		}

		template <AcType T>
		inline void setAllTo(const Packet<T, float> &setTo) {
			for (int i = 0; i < Tsize / Packet<T>::len; i++)
				get<T>(i) = setTo;
		}
		inline void setAllTo(float val) {
			maxPacket pack(val);
			setAllTo(pack);
		}
		inline PStore<1, Tlen, float> row(int i) { return PStore<1, Tlen, float>(CStorage::data + Tlen*i); }
	};

	template <int Tdim, int Tlen, typename BaseType>
	class PStore : public StoreBase<Tdim, Tlen, BaseType, PStorage<Tdim, Tlen, BaseType> > {
	public:
		typedef StoreBase<Tdim, Tlen, BaseType, PStorage<Tdim, Tlen, BaseType> > Base;
		inline PStore(BaseType* nDat) { Base::data = nDat; }
	};

	template <int Tdim, int Tlen, typename BaseType>
	class Store; // : public StoreBase<Tdim, Tlen, FStorage<Tdim, Tlen, BaseType> > {}; 

	template <int Tdim, typename BType> class
		Store<Tdim, 1, BType> : public StoreBase<Tdim, 1, BType, FStorage<Tdim, 1, BType> > {};

	template <int Tdim, typename BType> class
#ifdef _MSC_VER
		__declspec(align(16))
#endif
		Store<Tdim, 4, BType>: public StoreBase<Tdim, 4, BType, FStorage<Tdim, 4, BType> >{ public: MAKE_ALIGNED_OPERATOR_NEW };

	template <int Tdim, typename BType> class
#ifdef _MSC_VER
		__declspec(align(32))
#endif
		Store<Tdim, 8, BType>: public StoreBase<Tdim, 8, BType, FStorage<Tdim, 8, BType> >{ public: MAKE_ALIGNED_OPERATOR_NEW };

	template <int Tdim, typename BType> class
#ifdef _MSC_VER
		__declspec(align(32))
#endif
		Store<Tdim, 16, BType>: public StoreBase<Tdim, 16, BType, FStorage<Tdim, 16, BType> >{ public: MAKE_ALIGNED_OPERATOR_NEW };

	//These are the classes uses by the enduser
	template <int Tdim, int Tlen>
	class IStore : public Store<Tdim, Tlen, unsigned> { public: MAKE_ALIGNED_OPERATOR_NEW };

	template <int Tdim, int Tlen>
	class FStore : public Store<Tdim, Tlen, float> { public: MAKE_ALIGNED_OPERATOR_NEW };

	template <int Tdim, int Tlen>
	class IPStore : public PStore<Tdim, Tlen, unsigned> {};

	template <int Tdim, int Tlen>
	class FPStore : public PStore<Tdim, Tlen, float> {
	public:
		inline FPStore(float *fl) : PStore<Tdim, Tlen, float>(fl) {}
	};



	//OpFrom0
	template <AcType T, typename Accel> Packet<T, Accel> FAddInv();
	template <> inline Packet<AC_NONE, float> FAddInv<AC_NONE, float>() { return memcnvIF(0x80000000); }
	template <> inline Packet<AC_SSE, float> FAddInv<AC_SSE, float>() { return _mm_set1_ps(memcnvIF(0x80000000)); }
	template <> inline Packet<AC_SSE, double> FAddInv<AC_SSE, double>() { return _mm_set1_pd(memcnvID(0x8000'0000'0000'0000)); }

	//OpFrom1
	inline Packet<AC_NONE, float> FMulInv(const Packet<AC_NONE, float> &a) { return 1.0f / a.val; }
	inline Packet<AC_SSE, float> FMulInv(const Packet<AC_SSE, float> &a) { return _mm_div_ps(_mm_set1_ps(1), a); }
	inline Packet<AC_SSE, double> FMulInv(const Packet<AC_SSE, double>& a) { return _mm_div_pd(_mm_set1_pd(1), a); }

	inline Packet<AC_NONE, float> FAddIn(const Packet<AC_NONE, float> &a) { return Packet<AC_NONE, float>(-a.val); }
	inline Packet<AC_SSE, float> FAddInv(const Packet<AC_SSE, float> &a) { return _mm_xor_ps(_mm_set1_ps(memcnvIF(0x80000000)), a); }
	inline Packet<AC_NONE, unsigned> FAddInv(const Packet<AC_NONE, unsigned> &a) { return Packet<AC_NONE, unsigned>(-(signed)a.val); }
	inline Packet<AC_SSE, unsigned> FAddInv(const Packet<AC_SSE, unsigned> &a) { __m128i zero = _mm_setzero_si128(); return _mm_sub_epi32(zero, a.val); }
	inline Packet<AC_NONE, double> FAddInv(const Packet<AC_NONE, double>& a) { return Packet<AC_NONE, double>(-a.val); }
	inline Packet<AC_SSE, double> FAddInv(const Packet<AC_SSE, double>& a) { __m128d xorval = _mm_set1_pd(memcnvID(0x8000'0000'0000'0000)); return _mm_xor_pd(xorval, a.val); }

	inline Packet<AC_NONE, float> FSqrt(const Packet<AC_NONE, float>& a) { return sqrtf(a); }
	inline Packet<AC_NONE, double> FSqrt(const Packet<AC_NONE, double> &a) { return sqrt(a.val); }
	inline Packet<AC_SSE, float> FSqrt(const Packet<AC_SSE, float> &a) { return _mm_sqrt_ps(a); }
	inline Packet<AC_SSE, double> FSqrt(const Packet<AC_SSE, double>& a) { return _mm_sqrt_pd(a); }

	inline float FRowSum(const Packet<AC_NONE, float> &a) { return a; }
	inline float FRowSum(const Packet<AC_SSE, float> &a) {
		__m128 result = _mm_movehl_ps(a, a);
		__m128 tmp = _mm_add_ps(result, a);
		result = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 1, 1, 1));
		result = _mm_add_ps(result, tmp);
		Packet<AC_SSE, float> retEx(result); //reason: naming problem with gcc
		return ss_getf(retEx, 0);
	}

	inline Packet<AC_SSE, unsigned> FCastInt(const Packet<AC_SSE, float> &val) { return _mm_castps_si128(val); }
	inline Packet<AC_SSE, unsigned> FCnvInt(const Packet<AC_SSE, float> &val) { return _mm_cvtps_epi32(val); }
	inline Packet<AC_SSE, float> FCastFloat(const Packet<AC_SSE, unsigned> &val) { return _mm_castsi128_ps(val); }
	inline Packet<AC_SSE, float> FCnvFloat(const Packet<AC_SSE, unsigned> &val) { return _mm_cvtepi32_ps(val); }


	//OpFrom2
	
	inline Packet<AC_NONE, double> FAdd(const Packet<AC_NONE, double>& a, const Packet<AC_NONE, double>& b) { return (a.val + b.val); }//Elementrary
	inline Packet<AC_NONE, float> FAdd(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float> &b) { return (a.val + b.val); }
	inline Packet<AC_SSE, float> FAdd(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float> &b) { return _mm_add_ps(a, b); }
	inline Packet<AC_SSE, double> FAdd(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_add_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator+(const Packet<T, TY> &a, const Packet<T, TY>& b)
	{
		return FAdd(a, b);
	}

	inline Packet<AC_NONE, float> FMul(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float> &b) { return a*b; }
	inline Packet<AC_NONE, double> FMul(const Packet<AC_NONE, double>& a, const Packet<AC_NONE, double>& b) { return a.val * b.val; }
	inline Packet<AC_SSE, float> FMul(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float> &b) { return _mm_mul_ps(a, b); }
	inline Packet<AC_SSE, double> FMul(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_mul_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator*(const Packet<T, TY> &a, const Packet<T, TY> &b)
	{
		return FMul(a, b);
	}

	inline Packet<AC_NONE, float> FSub(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float> &b) { return a.val - b.val; }
	inline Packet<AC_NONE, double> FSub(const Packet<AC_NONE, double>& a, const Packet<AC_NONE, double>& b) { return a.val - b.val; }
	inline Packet<AC_SSE, float> FSub(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float> &b) { return _mm_sub_ps(a, b); }
	inline Packet<AC_SSE, double> FSub(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_sub_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator-(const Packet<T, TY> &a, const Packet<T, TY> &b)
	{
		return FSub(a, b);
	}

	inline Packet<AC_NONE, float> FDiv(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float> &b) { return a.val / b.val; }
	inline Packet<AC_NONE, double> FDiv(const Packet<AC_NONE, double>& a, const Packet<AC_NONE, double>& b) { return a.val / b.val; }
	inline Packet<AC_SSE, float> FDiv(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float> &b) { return _mm_div_ps(a, b); }
	inline Packet<AC_SSE, double> FDiv(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_div_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator/(const Packet<T, TY> &a, const Packet<T, TY> &b)
	{
		return FDiv(a, b);
	}

	template <AcType T, typename TY, unsigned SHFV> inline Packet<T, TY> FShuffle(const Packet<T, TY> &a, const Packet<T, TY> &b)
	{
		return _mm_shuffle_ps(a, b, SHFV);
	}


	//Moves
	inline Packet<AC_NONE, float> FMlh(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& /*b*/) { return a; }
	inline Packet<AC_SSE, float> FMlh(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_movelh_ps(a, b); }

	inline Packet<AC_NONE, float> FMhl(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& /*b*/) { return a; }
	inline Packet<AC_SSE, float> FMhl(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_movehl_ps(a, b); }

	//Logical
	inline Packet<AC_NONE, float> FOr(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return memcnvIF(memcnvFI(a) | memcnvFI(b)); }
	inline Packet<AC_SSE, float> FOr(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_or_ps(a, b); }
	inline Packet<AC_SSE, unsigned> FOr(const Packet<AC_SSE, unsigned> &a, const Packet<AC_SSE, unsigned>& b) { return _mm_or_si128(a, b); }
	inline Packet<AC_SSE, double> FOr(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_or_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator|(const Packet<T, TY> &a, const Packet<T, TY>& b)
	{
		return FOr(a, b);
	}

	inline Packet<AC_NONE, float> FXor(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return memcnvIF(memcnvFI(a) ^ memcnvFI(b)); }
	inline Packet<AC_SSE, float> FXor(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_xor_ps(a, b); }
	inline Packet<AC_SSE, unsigned> FXor(const Packet<AC_SSE, unsigned> &a, const Packet<AC_SSE, unsigned>& b) { return _mm_xor_si128(a, b); }
	inline Packet<AC_SSE, double> FXor(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_xor_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator^(const Packet<T, TY> &a, const Packet<T, TY>& b)
	{
		return FXor(a, b);
	}

	inline Packet<AC_NONE, float> FAnd(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return memcnvIF(memcnvFI(a) & memcnvFI(b)); }
	inline Packet<AC_SSE, float> FAnd(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_and_ps(a, b); }
	inline Packet<AC_SSE, unsigned> FAnd(const Packet<AC_SSE, unsigned> &a, const Packet<AC_SSE, unsigned>& b) { return _mm_and_si128(a, b); }
	inline Packet<AC_SSE, double> FAnd(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_and_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator&(const Packet<T, TY> &a, const Packet<T, TY>& b)
	{
		return FAnd(a, b);
	}

	inline Packet<AC_NONE, float> FAndNot(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return memcnvIF((~memcnvFI(a)) & memcnvFI(b)); }
	inline Packet<AC_SSE, float> FAndNot(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_andnot_ps(a, b); }
	inline Packet<AC_SSE, unsigned> FAndNot(const Packet<AC_SSE, unsigned> &a, const Packet<AC_SSE, unsigned>& b) { return _mm_andnot_si128(a, b); }
	inline Packet<AC_SSE, double> FAndNot(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_andnot_pd(a, b); }

	//Comparisons
	inline Packet<AC_NONE, float> FCmpEq(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return a == b ? memcnvFull() : memcnvZero(); }
	inline Packet<AC_SSE, float> FCmpEq(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_cmpeq_ps(a, b); }
	inline Packet<AC_SSE, double> FCmpEq(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_cmpeq_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator==(const Packet<T, TY> &a, const Packet<T, TY>& b) { return FCmpEq(a, b);	}

	inline Packet<AC_NONE, float> FCmpNeq(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return !(a == b) ? memcnvFull() : memcnvZero(); }
	inline Packet<AC_SSE, float> FCmpNeq(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_cmpneq_ps(a, b); }
	inline Packet<AC_SSE, double> FCmpNeq(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_cmpneq_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator!=(const Packet<T, TY> &a, const Packet<T, TY>& b)	{ return FCmpNeq(a, b); }

	inline Packet<AC_NONE, float> FCmpLt(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return a < b ? memcnvFull() : memcnvZero(); }
	inline Packet<AC_SSE, float> FCmpLt(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_cmplt_ps(a, b); }
	inline Packet<AC_SSE, double> FCmpLt(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_cmplt_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator<(const Packet<T, TY> &a, const Packet<T, TY>& b) { return FCmpLt(a, b); }

	inline Packet<AC_NONE, float> FCmpLe(const Packet<AC_NONE, float> &a, const Packet<AC_NONE, float>& b) { return a <= b ? memcnvFull() : memcnvZero(); }
	inline Packet<AC_SSE, float> FCmpLe(const Packet<AC_SSE, float> &a, const Packet<AC_SSE, float>& b) { return _mm_cmple_ps(a, b); }
	inline Packet<AC_SSE, double> FCmpLe(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_cmple_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator<=(const Packet<T, TY> &a, const Packet<T, TY>& b) { return FCmpLe(a, b); }

	inline Packet<AC_NONE, float> FCmpGt(const Packet<AC_NONE, float>& a, const Packet<AC_NONE, float>& b) { return a > b ? memcnvFull() : memcnvZero(); }
	inline Packet<AC_NONE, double> FCmpGt(const Packet<AC_NONE, double>& a, const Packet<AC_NONE, double>& b) { return a.val > b.val ? memcnvFullD() : 0.0; }
	inline Packet<AC_SSE, float> FCmpGt(const Packet<AC_SSE, float>& a, const Packet<AC_SSE, float>& b) { return _mm_cmpgt_ps(a, b); }
	inline Packet<AC_SSE, double> FCmpGt(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_cmpgt_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator>(const Packet<T, TY>& a, const Packet<T, TY>& b)	{ return FCmpGt(a, b); }

	inline Packet<AC_NONE, float> FCmpGe(const Packet<AC_NONE, float>& a, const Packet<AC_NONE, float>& b) { return a >= b ? memcnvFull() : memcnvZero(); }
	inline Packet<AC_SSE, float> FCmpGe(const Packet<AC_SSE, float>& a, const Packet<AC_SSE, float>& b) { return _mm_cmpnlt_ps(a, b); }
	inline Packet<AC_SSE, double> FCmpGe(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_cmpnlt_pd(a, b); }
	template <AcType T, typename TY>
	inline Packet<T, TY> operator>=(const Packet<T, TY>& a, const Packet<T, TY>& b) { return FCmpGe(a, b);	}

	inline Packet<AC_NONE, float> FMin(const Packet<AC_NONE, float>& a, const Packet<AC_NONE, float>& b) { return a <= b ? a : b; }
	inline Packet<AC_SSE, float> FMin(const Packet<AC_SSE, float>& a, const Packet<AC_SSE, float>& b) { return _mm_min_ps(a, b); }
	inline Packet<AC_SSE, double> FMin(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_min_pd(a, b); }
	inline Packet<AC_SSE, unsigned> FMin(const Packet<AC_SSE, unsigned>& a, const Packet<AC_SSE, unsigned>& b) { return _mm_min_epi32(a, b); }

	inline Packet<AC_NONE, float> FMax(const Packet<AC_NONE, float>& a, const Packet<AC_NONE, float>& b) { return a >= b ? a : b; }
	inline Packet<AC_NONE, double> FMax(const Packet<AC_NONE, double>& a, const Packet<AC_NONE, double>& b) { return a.val >= b.val ? a.val : b.val; }
	inline Packet<AC_SSE, float> FMax(const Packet<AC_SSE, float>& a, const Packet<AC_SSE, float>& b) { return _mm_max_ps(a, b); }
	inline Packet<AC_SSE, double> FMax(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b) { return _mm_max_pd(a, b); }
	inline Packet<AC_SSE, unsigned> FMax(const Packet<AC_SSE, unsigned>& a, const Packet<AC_SSE, unsigned>& b) { return _mm_max_epi32(a, b); }

	//OpFrom3
	//a ? b : c
	inline Packet<AC_NONE, float> FBinSelect(const Packet<AC_NONE, float>& a, const Packet<AC_NONE, float>& b, const Packet<AC_NONE, float>& c) { return memcnvIF((memcnvFI(a)&memcnvFI(b)) | ((~memcnvFI(a))&memcnvFI(c))); }
	inline Packet<AC_NONE, double> FBinSelect(const Packet<AC_NONE, double>& a, const Packet<AC_NONE, double>& b, const Packet<AC_NONE, double>& c) { return memcnvID((memcnvDI(a.val) & memcnvDI(b.val)) | ((~memcnvDI(a.val)) & memcnvDI(c.val))); }
	inline Packet<AC_SSE, float> FBinSelect(const Packet<AC_SSE, float>& a, const Packet<AC_SSE, float>& b, const Packet<AC_SSE, float>& c) { return _mm_or_ps(_mm_and_ps(a, b), _mm_andnot_ps(a, c)); }
	inline Packet<AC_SSE, double> FBinSelect(const Packet<AC_SSE, double>& a, const Packet<AC_SSE, double>& b, const Packet<AC_SSE, double>& c) { return _mm_or_pd(_mm_and_pd(a, b), _mm_andnot_pd(a, c)); }
	inline Packet<AC_NONE, unsigned> FBinSelect(const Packet<AC_NONE, unsigned>& a, const Packet<AC_NONE, unsigned>& b, const Packet<AC_NONE, unsigned>& c) { return (a.val&b.val) | ((~a.val)&c.val); }
	inline Packet<AC_SSE, unsigned> FBinSelect(const Packet<AC_SSE, unsigned>& a, const Packet<AC_SSE, unsigned>& b, const Packet<AC_SSE, unsigned>& c) { return (a & b) | FAndNot(a, c); }

#ifdef INCLUDE_AVX
	//--------------------------------------------------------- SINGLE PRECISION --------------------------------------------------

	template <> inline Packet<AC_AVX, float> FAddInv<AC_AVX, float>() { return _mm256_set1_ps(memcnvIF(0x80000000)); }
	inline Packet<AC_AVX, float> FOr(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_or_ps(a, b); }
	inline Packet<AC_AVX, float> FDiv(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_div_ps(a, b); }
	inline Packet<AC_AVX, float> FSub(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_sub_ps(a, b); }
	inline Packet<AC_AVX, float> FMul(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_mul_ps(a, b); }
	inline Packet<AC_AVX, float> FAdd(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_add_ps(a, b); }
	inline float FRowSum(const Packet<AC_AVX, float>& a) {
		__m256 lowpart = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 0, 1)); //shuffles quadwords separately
		__m256 sum1 = _mm256_add_ps(a, lowpart);
		__m256 shuf2 = _mm256_shuffle_ps(sum1, sum1, _MM_SHUFFLE(3, 3, 3, 3));
		__m256 sum2 = _mm256_add_ps(shuf2, sum1); // 2,3 + 6,7: are valid 
		__m256 low4 = _mm256_permute2f128_ps(sum2, sum2, 1);
		__m256 result = _mm256_add_ps(low4, sum2);
		Packet<AC_AVX, float> retEx(result);
		return ss_getf(retEx, 3);
	}
	inline Packet<AC_AVX, float> FSqrt(const Packet<AC_AVX, float>& a) { return _mm256_sqrt_ps(a); }
	inline Packet<AC_AVX, float> FAddInv(const Packet<AC_AVX, float>& a) { return _mm256_xor_ps(_mm256_set1_ps(memcnvIF(0x80000000)), a); }
	inline Packet<AC_AVX, float> FMulInv(const Packet<AC_AVX, float>& a) { return _mm256_div_ps(_mm256_set1_ps(1), a); }
	inline Packet<AC_AVX, float> FCmpLt(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(a, b, 1); }
	inline Packet<AC_AVX, float> FCmpGt(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(b, a, 1); }
	inline Packet<AC_AVX, float> FCmpLe(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(a, b, 2); }
	inline Packet<AC_AVX, float> FCmpGe(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(b, a, 2); }
	inline Packet<AC_AVX, float> FCmpNlt(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(a, b, 5); }
	inline Packet<AC_AVX, float> FCmpNgt(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(b, a, 5); }
	inline Packet<AC_AVX, float> FCmpNeq(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(a, b, 4); }
	inline Packet<AC_AVX, float> FCmpEq(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_cmp_ps(a, b, 0); }
	inline Packet<AC_AVX, float> FAnd(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_and_ps(a, b); }
	inline Packet<AC_AVX, float> FXor(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_xor_ps(a, b); }
	inline Packet<AC_AVX, float> FAndNot(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_andnot_ps(a, b); }

	inline Packet<AC_AVX, float> FMin(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_min_ps(a, b); }
	inline Packet<AC_AVX, float> FMax(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return _mm256_max_ps(a, b); }
	inline Packet<AC_AVX, float> FBinSelect(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b, const Packet<AC_AVX, float>& c) { return _mm256_or_ps(_mm256_and_ps(a, b), _mm256_andnot_ps(a, c)); }

	inline Packet<AC_AVX, float> operator==(const Packet<AC_AVX, float> &a, const Packet<AC_AVX, float>& b) { return FCmpEq(a, b); }
	inline Packet<AC_AVX, float> operator!=(const Packet<AC_AVX, float> &a, const Packet<AC_AVX, float>& b) { return FCmpNeq(a, b); }
	inline Packet<AC_AVX, float> operator<(const Packet<AC_AVX, float> &a, const Packet<AC_AVX, float>& b) { return FCmpLt(a, b); }
	inline Packet<AC_AVX, float> operator<=(const Packet<AC_AVX, float> &a, const Packet<AC_AVX, float>& b) { return FCmpLe(a, b); }
	inline Packet<AC_AVX, float> operator>(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return FCmpGt(a, b); }
	inline Packet<AC_AVX, float> operator>=(const Packet<AC_AVX, float>& a, const Packet<AC_AVX, float>& b) { return FCmpGe(a, b); }

//--------------------------------------------------------- DOUBLE PRECISION --------------------------------------------------

	inline Packet<AC_AVX, double> FOr(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_or_pd(a, b); }
	inline Packet<AC_AVX, double> FDiv(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_div_pd(a, b); }
	inline Packet<AC_AVX, double> FSub(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_sub_pd(a, b); }
	inline Packet<AC_AVX, double> FMul(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_mul_pd(a, b); }
	inline Packet<AC_AVX, double> FAdd(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_add_pd(a, b); }
	inline double FRowSum(const Packet<AC_AVX, double>& a) {
		//TODO: this can be made more efficient with intrinsics
		double sum = 0;
		for (int k = 0; k < 4; k++)
			sum += a[k];
		return sum;
	}
	inline Packet<AC_AVX, double> FSqrt(const Packet<AC_AVX, double>& a) { return _mm256_sqrt_pd(a); }
	inline Packet<AC_AVX, double> FAddInv(const Packet<AC_AVX, double>& a) { return _mm256_xor_pd(_mm256_set1_pd(memcnvID(0x8000'0000'0000'0000)), a); }
	inline Packet<AC_AVX, double> FMulInv(const Packet<AC_AVX, double>& a) { return _mm256_div_pd(_mm256_set1_pd(1), a); }
	inline Packet<AC_AVX, double> FCmpLt(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(a, b, 1); }
	inline Packet<AC_AVX, double> FCmpGt(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(b, a, 1); }
	inline Packet<AC_AVX, double> FCmpLe(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(a, b, 2); }
	inline Packet<AC_AVX, double> FCmpGe(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(b, a, 2); }
	inline Packet<AC_AVX, double> FCmpNlt(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(a, b, 5); }
	inline Packet<AC_AVX, double> FCmpNgt(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(b, a, 5); }
	inline Packet<AC_AVX, double> FCmpNeq(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(a, b, 4); }
	inline Packet<AC_AVX, double> FCmpEq(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_cmp_pd(a, b, 0); }
	inline Packet<AC_AVX, double> FAnd(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_and_pd(a, b); }
	inline Packet<AC_AVX, double> FXor(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_xor_pd(a, b); }
	inline Packet<AC_AVX, double> FAndNot(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_andnot_pd(a, b); }

	inline Packet<AC_AVX, double> FMin(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_min_pd(a, b); }
	inline Packet<AC_AVX, double> FMax(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return _mm256_max_pd(a, b); }
	inline Packet<AC_AVX, double> FBinSelect(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b, const Packet<AC_AVX, double>& c) { return _mm256_or_pd(_mm256_and_pd(a, b), _mm256_andnot_pd(a, c)); }

	inline Packet<AC_AVX, double> operator==(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return FCmpEq(a, b); }
	inline Packet<AC_AVX, double> operator!=(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return FCmpNeq(a, b); }
	inline Packet<AC_AVX, double> operator<(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return FCmpLt(a, b); }
	inline Packet<AC_AVX, double> operator<=(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return FCmpLe(a, b); }
	inline Packet<AC_AVX, double> operator>(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return FCmpGt(a, b); }
	inline Packet<AC_AVX, double> operator>=(const Packet<AC_AVX, double>& a, const Packet<AC_AVX, double>& b) { return FCmpGe(a, b); }
#endif


#ifdef INCLUDE_AVX512
	template <> inline Packet<AC_AVX512, float> FAddInv<AC_AVX512, float>() { return _mm512_set1_ps(memcnvIF(0x80000000)); }
	inline Packet<AC_AVX512, float> FOr(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_or_ps(a, b); }
	inline Packet<AC_AVX512, float> FDiv(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_div_ps(a, b); }
	inline Packet<AC_AVX512, float> FSub(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_sub_ps(a, b); }
	inline Packet<AC_AVX512, float> FMul(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_mul_ps(a, b); }
	inline Packet<AC_AVX512, float> FAdd(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_add_ps(a, b); }
	//inline float FRowSum(const Packet<AC_AVX512, float>& a) {
	//	__m256 lowpart = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 0, 1)); //shuffles quadwords separately
	//	__m256 sum1 = _mm256_add_ps(a, lowpart);
	//	__m256 shuf2 = _mm256_shuffle_ps(sum1, sum1, _MM_SHUFFLE(3, 3, 3, 3));
	//	__m256 sum2 = _mm256_add_ps(shuf2, sum1); // 2,3 + 6,7: are valid 
	//	__m256 low4 = _mm256_permute2f128_ps(sum2, sum2, 1);
	//	__m256 result = _mm256_add_ps(low4, sum2);
	//	Packet<AC_AVX, float> retEx(result);
	//	return ss_getf(retEx, 3);
	//}
	inline Packet<AC_AVX512, float> FSqrt(const Packet<AC_AVX512, float>& a) { return _mm512_sqrt_ps(a); }
	inline Packet<AC_AVX512, float> FAddInv(const Packet<AC_AVX512, float>& a) { return _mm512_xor_ps(_mm512_set1_ps(memcnvIF(0x80000000)), a); }
	inline Packet<AC_AVX512, float> FMulInv(const Packet<AC_AVX512, float>& a) { return _mm512_div_ps(_mm512_set1_ps(1), a); }
	inline AVX512MASK_16 FCmpLt(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmplt_ps_mask(a, b); }
	inline AVX512MASK_16 FCmpGt(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmp_ps_mask (b, a, 1); }
	inline AVX512MASK_16 FCmpLe(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmple_ps_mask(a, b); }
	inline AVX512MASK_16 FCmpGe(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmpnlt_ps_mask(b, a); }
	inline AVX512MASK_16 FCmpNlt(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmp_ps_mask (a, b, 5); }
	inline AVX512MASK_16 FCmpNgt(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmp_ps_mask (b, a, 5); }
	inline AVX512MASK_16 FCmpNeq(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmp_ps_mask (a, b, 4); }
	inline AVX512MASK_16 FCmpEq(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_cmpeq_ps_mask(a, b); }
	inline Packet<AC_AVX512, float> FAnd(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_and_ps(a, b); }
	inline Packet<AC_AVX512, float> FXor(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_xor_ps(a, b); }
	inline Packet<AC_AVX512, float> FAndNot(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_andnot_ps(a, b); }

	inline Packet<AC_AVX512, float> FMin(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_min_ps(a, b); }
	inline Packet<AC_AVX512, float> FMax(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return _mm512_max_ps(a, b); }
	//TODO: use masks here
	inline Packet<AC_AVX512, float> FBinSelect(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b, const Packet<AC_AVX512, float>& c) 
		{ return _mm512_or_ps(_mm512_and_ps(a, b), _mm512_andnot_ps(a, c)); }

	inline AVX512MASK_16 operator==(const Packet<AC_AVX512, float> &a, const Packet<AC_AVX512, float>& b) { return FCmpEq(a, b); }
	inline AVX512MASK_16 operator!=(const Packet<AC_AVX512, float> &a, const Packet<AC_AVX512, float>& b) { return FCmpNeq(a, b); }
	inline AVX512MASK_16 operator<(const Packet<AC_AVX512, float> &a, const Packet<AC_AVX512, float>& b) { return FCmpLt(a, b); }
	inline AVX512MASK_16 operator<=(const Packet<AC_AVX512, float> &a, const Packet<AC_AVX512, float>& b) { return FCmpLe(a, b); }
	inline AVX512MASK_16 operator>(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return FCmpGt(a, b); }
	inline AVX512MASK_16 operator>=(const Packet<AC_AVX512, float>& a, const Packet<AC_AVX512, float>& b) { return FCmpGe(a, b); }
#endif
}
