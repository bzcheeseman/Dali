#ifndef DALI_ARRAY_REDUCER_KERNELS_H
#define DALI_ARRAY_REDUCER_KERNELS_H

#include <climits>
#include <cfloat>
/*! \brief namespace for potential reducer operations.
 *  (copied over from mshadow awaiting customization)
 */
namespace reducers {
    namespace limits {
        /*!
         * \brief minimum value of certain types
         * \tparam DType data type
         */
        template<typename DType>
        XINLINE DType MinValue(void);
        /*! \brief minimum value of float */
        template<>
        XINLINE float MinValue<float>(void) {
            return -FLT_MAX;
        }
        /*! \brief minimum value of double */
        template<>
        XINLINE double MinValue<double>(void) {
            return -DBL_MAX;
        }
        /*! \brief minimum value of int */
        template<>
        XINLINE int MinValue<int>(void) {
            return INT_MIN;
        }

        /*!
         * \brief maximum value of certain types
         * \tparam DType data type
         */
        template<typename DType>
        XINLINE DType MaxValue(void);
        /*! \brief maximum value of float */
        template<>
        XINLINE float MaxValue<float>(void) {
            return FLT_MAX;
        }
        /*! \brief maximum value of double */
        template<>
        XINLINE double MaxValue<double>(void) {
            return DBL_MAX;
        }
        /*! \brief maximum value of int */
        template<>
        XINLINE int MaxValue<int>(void) {
            return INT_MAX;
        }

    }  // namespace limits

    /*! \brief sum reducer */
    struct sum {
        static const bool reduce_by_pool_area = false;

        /*! \brief do reduction into dst */
        template<typename DType>
        XINLINE static void Reduce(volatile DType& dst,  volatile DType src) {
            dst += src;
        }
        /*!
         *\brief calculate gradient of redres with respect to redsrc,
         * redres: reduced result, redsrc: one of reduction element
         */
        template<typename DType>
        XINLINE static DType PartialGrad(DType redres, DType redsrc) {
            return 1;
        }
        /*!
         *\brief set the initial value during reduction
         */
        template<typename DType>
        XINLINE static void SetInitValue(DType &initv) {
            initv = 0;
        }
    };

    struct avg : sum {
        static const bool reduce_by_pool_area = true;
    };

    /*! \brief maximum reducer */
    struct maximum {
        static const bool reduce_by_pool_area = false;
        /*! \brief do reduction into dst */
        template<typename DType>
        XINLINE static void Reduce(volatile DType& dst,  volatile DType src) {
            dst = dst > src ? dst : src;
        }
        XINLINE static void Reduce(volatile float& dst,  volatile float src) {
            #ifdef __CUDACC__
                dst = fmaxf(dst, src);
            #else
                dst = dst > src ? dst : src;
            #endif
        }
        /*!
         * \brief calculate gradient of redres with respect to redsrc,
         * redres: reduced result, redsrc: one of reduction element
         */
        template<typename DType>
        XINLINE static DType PartialGrad(DType redres, DType redsrc) {
            return redres == redsrc ? 1: 0;
        }
        /*!
         *\brief set the initial value during reduction
         */
        template<typename DType>
        XINLINE static void SetInitValue(DType &initv) {
            initv = limits::MinValue<DType>();
        }
    };
    /*! \brief minimum reducer */
    struct minimum {
        static const bool reduce_by_pool_area = false;
        /*! \brief do reduction into dst */
        template<typename DType>
        XINLINE static void Reduce(volatile DType& dst,  volatile DType src) {
    #ifdef __CUDACC__
            dst = ::min(dst, src);
    #else
            dst = dst < src ? dst : src;
    #endif  // __CUDACC__
        }
        /*!
         * \brief calculate gradient of redres with respect to redsrc,
         * redres: reduced result, redsrc: one of reduction element
         */
        template<typename DType>
        XINLINE static DType PartialGrad(DType redres, DType redsrc) {
            return redres == redsrc ? 1: 0;
        }
        /*!
         *\brief set the initial value during reduction
         */
        template<typename DType>
        XINLINE static void SetInitValue(DType &initv) {
            initv = limits::MaxValue<DType>();
        }
    };
    /*! \brief product reducer */
    struct product {
        static const bool reduce_by_pool_area = false;
        /*! \brief do reduction into dst */
        template<typename DType>
        XINLINE static void Reduce(volatile DType& dst,  volatile DType src) {
            dst *= src;
        }
        /*!
         * \brief calculate gradient of redres with respect to redsrc,
         * redres: reduced result, redsrc: one of reduction element
         */
        template<typename DType>
        XINLINE static DType PartialGrad(DType redres, DType redsrc) {
            return redsrc != 0 ? redres / redsrc : 0;
        }
        /*!
         *\brief set the initial value during reduction
         */
        template<typename DType>
        XINLINE static void SetInitValue(DType &initv) {
            initv = 1.0;
        }
    };
}  // namespace reducers

#endif  // DALI_ARRAY_REDUCER_KERNELS_H
