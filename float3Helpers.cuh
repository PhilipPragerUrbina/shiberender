#pragma once

#include "cuda_runtime.h"



#include <math.h>





__device__
inline float3 getCrossProduct(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__
inline float3 make3(float a)
{
    return make_float3(a, a, a);
}
__device__
inline float4 getCrossProduct(float4 a, float4 b)
{
    float3 v1 = make_float3(a.x, a.y, a.z);
    float3 v2 = make_float3(b.x, b.y, b.z);
    float3 v3 = make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

    return make_float4(v3.x, v3.y, v3.z, 0.0f);
}

__host__ __device__
inline float getDotProduct(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float getDotProduct(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ float3 getNormalizedVec(const float3 v)
{
    float invLen = 1.0f / sqrtf(getDotProduct(v, v));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ float4 getNormalizedVec(const float4 v)
{
    float invLen = 1.0f / sqrtf(getDotProduct(v, v));
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, v.w * invLen);
}

__device__
inline float dot3F4(float4 a, float4 b)
{
    float4 a1 = make_float4(a.x, a.y, a.z, 0.f);
    float4 b1 = make_float4(b.x, b.y, b.z, 0.f);
    return getDotProduct(a1, b1);
}

__host__ __device__
inline float getLength(float3 a)
{
    return sqrtf(getDotProduct(a, a));
}


__device__
inline float getLength(float4 a)
{
    return sqrtf(getDotProduct(a, a));
}
//vector operators
__host__ __device__  float3 operator+(const float3& a, const float3& b) {

    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}
__host__ __device__  float3 operator-(const float3& a, const float3& b) {

    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}
__host__ __device__  float3 operator*(const float3& a, const float3& b) {

    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);

}
__host__ __device__  float3 operator/(const float3& a, const float3& b) {

    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);

}

typedef struct
{
    float4 m_row[4];
}Matrix3x3_d;

__device__
inline void setZero(Matrix3x3_d& m)
{
    m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__
inline Matrix3x3_d getZeroMatrix3x3()
{
    Matrix3x3_d m;
    m.m_row[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    m.m_row[3] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return m;
}

__device__
inline void setIdentity(Matrix3x3_d& m)
{
    m.m_row[0] = make_float4(1, 0, 0, 0);
    m.m_row[1] = make_float4(0, 1, 0, 0);
    m.m_row[2] = make_float4(0, 0, 1, 0);
    m.m_row[3] = make_float4(0, 0, 0, 1);
}

__device__
inline Matrix3x3_d getIdentityMatrix3x3()
{
    Matrix3x3_d m;
    m.m_row[0] = make_float4(1, 0, 0, 0);
    m.m_row[1] = make_float4(0, 1, 0, 0);
    m.m_row[2] = make_float4(0, 0, 1, 0);
    m.m_row[3] = make_float4(0, 0, 0, 1);
    return m;
}

__device__
inline Matrix3x3_d getTranspose(const Matrix3x3_d m)
{
    Matrix3x3_d out;
    out.m_row[0] = make_float4(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.f);
    out.m_row[1] = make_float4(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.f);
    out.m_row[2] = make_float4(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.f);
    return out;
}
__device__ float3 multVecMatrix(Matrix3x3_d x, float3 src)
{
    float3 dst;
    float a, b, c, w;

    a = src.x * x.m_row[0].x + src.y * x.m_row[1].x + src.z * x.m_row[2].x + x.m_row[3].x;
    b = src.x * x.m_row[0].y + src.y * x.m_row[1].y + src.z * x.m_row[2].y + x.m_row[3].y;
    c = src.x * x.m_row[0].z + src.y * x.m_row[1].z + src.z * x.m_row[2].z + x.m_row[3].z;
    w = src.x * x.m_row[0].w + src.y * x.m_row[1].w + src.z * x.m_row[2].w + x.m_row[3].w;

    dst.x = a / w;
    dst.y = b / w;
    dst.z = c / w;
    return dst;
}

__device__ float distance(float3 p1, float3 p2)
{
    float diffY = p1.y - p2.y;
    float diffX = p1.x - p2.x;
    return sqrt((diffY * diffY) + (diffX * diffX));
}
__device__
 Matrix3x3_d MatrixMul(Matrix3x3_d& a, Matrix3x3_d& b)
{
    Matrix3x3_d transB = getTranspose(b);
    Matrix3x3_d ans;
    //        why this doesn't run when 0ing in the for{}
    a.m_row[0].w = 0.f;
    a.m_row[1].w = 0.f;
    a.m_row[2].w = 0.f;
    for (int i = 0; i < 3; i++)
    {
        //        a.m_row[i].w = 0.f;
        ans.m_row[i].x = dot3F4(a.m_row[i], transB.m_row[0]);
        ans.m_row[i].y = dot3F4(a.m_row[i], transB.m_row[1]);
        ans.m_row[i].z = dot3F4(a.m_row[i], transB.m_row[2]);
        ans.m_row[i].w = 0.f;
    }
    return ans;
}
__device__ float getFromIndex(float4 what, int index) {

    if (index == 0) {
        return what.x;

    }
    else if (index == 1) {

        return what.y;

    }
    else if (index == 2) {

        return what.z;

    }
    else {

        return what.w;

    }


}


__device__ void setFromIndex(float4& what, int index, float set) {

    if (index == 0) {
        what.x = set;

    }
    else if (index == 1) {

        what.y = set;

    }
    else if (index == 2) {

        what.z = set;

    }
    else {

        what.w = set;

    }


}
__device__ Matrix3x3_d inverse(Matrix3x3_d t)
{
    int i, j, k;
    Matrix3x3_d s;
    setIdentity(s);

    // Forward elimination
    for (i = 0; i < 3; i++) {
        int pivot = i;

        float pivotsize = getFromIndex(t.m_row[i], i);

        if (pivotsize < 0) {

            pivotsize = -pivotsize;
        }


        for (j = i + 1; j < 4; j++) {
            float tmp = getFromIndex(t.m_row[j], i);

            if (tmp < 0) {
                tmp = -tmp;

            }


            if (tmp > pivotsize) {
                pivot = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0) {
            // Cannot invert singular matrix

            return getIdentityMatrix3x3();
        }

        if (pivot != i) {
            for (j = 0; j < 4; j++) {
                float tmp;

                tmp = getFromIndex(t.m_row[i], j);
                setFromIndex(t.m_row[i], j, getFromIndex(t.m_row[pivot], j));
                setFromIndex(t.m_row[pivot], j, tmp);

                tmp = getFromIndex(s.m_row[i], j);
                setFromIndex(s.m_row[i], j, getFromIndex(s.m_row[pivot], j));
                setFromIndex(s.m_row[pivot], j, tmp);
            }
        }

        for (j = i + 1; j < 4; j++) {
            float f = getFromIndex(t.m_row[j], i) / getFromIndex(t.m_row[i], i);

            for (k = 0; k < 4; k++) {
                setFromIndex(t.m_row[j], k, getFromIndex(t.m_row[j], k) - (f * getFromIndex(t.m_row[i], k)));
                setFromIndex(s.m_row[j], k, getFromIndex(s.m_row[j], k) - (f * getFromIndex(s.m_row[i], k)));
            }
        }
    }


    // Backward substitution
    for (i = 3; i >= 0; --i) {
        float f;

        if ((f = getFromIndex(t.m_row[i], i)) == 0) {
            // Cannot invert singular matrix
            return getIdentityMatrix3x3();
        }

        for (j = 0; j < 4; j++) {
            setFromIndex(t.m_row[i], j, getFromIndex(t.m_row[i], j) / (f));
            setFromIndex(s.m_row[i], j, getFromIndex(s.m_row[i], j) / (f));
        }

        for (j = 0; j < i; j++) {
            f = getFromIndex(t.m_row[j], i);

            for (k = 0; k < 4; k++) {
                setFromIndex(t.m_row[j], k, getFromIndex(t.m_row[j], k) - (f * getFromIndex(t.m_row[i], k)));
                setFromIndex(s.m_row[j], k, getFromIndex(s.m_row[j], k) - (f * getFromIndex(s.m_row[i], k)));
            }
        }
    }

    return s;
}
