//Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_16_FP32_FWD_PTX_HPP
#define CUFFTDX_FFT_16_FP32_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<25, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<240>;
.reg .b64 rd<2>;
add.f32 f65, %32, %53;
add.f32 f66, %33, %55;
sub.f32 f67, %32, %53;
sub.f32 f68, %33, %55;
add.f32 f69, %42, %64;
add.f32 f70, %44, %65;
sub.f32 f71, %42, %64;
sub.f32 f72, %44, %65;
add.f32 f73, f65, f69;
add.f32 f74, f66, f70;
sub.f32 f75, f65, f69;
sub.f32 f76, f66, f70;
add.f32 f77, f67, f72;
sub.f32 f78, f68, f71;
sub.f32 f79, f67, f72;
add.f32 f80, f68, f71;
add.f32 f81, %37, %58;
add.f32 f82, %39, %60;
sub.f32 f83, %37, %58;
sub.f32 f84, %39, %60;
add.f32 f85, %48, %69;
add.f32 f86, %49, %71;
sub.f32 f87, %48, %69;
sub.f32 f88, %49, %71;
add.f32 f89, f81, f85;
add.f32 f90, f82, f86;
sub.f32 f91, f81, f85;
sub.f32 f92, f82, f86;
add.f32 f93, f83, f88;
sub.f32 f94, f84, f87;
sub.f32 f95, f83, f88;
add.f32 f96, f84, f87;
mul.f32 f97, f93, 0f3F3504F3;
mul.f32 f98, f94, 0fBF3504F3;
sub.f32 f99, f97, f98;
mul.f32 f100, f94, 0f3F3504F3;
fma.rn.f32 f101, f93, 0fBF3504F3, f100;
mul.f32 f102, f95, 0fBF3504F3;
mul.f32 f103, f96, 0fBF3504F3;
sub.f32 f104, f102, f103;
add.f32 f105, f102, f103;
add.f32 f106, f73, f89;
add.f32 f107, f74, f90;
sub.f32 f108, f73, f89;
sub.f32 f109, f74, f90;
add.f32 f110, f77, f99;
add.f32 f111, f78, f101;
sub.f32 f112, f77, f99;
sub.f32 f113, f78, f101;
add.f32 f114, f75, f92;
sub.f32 f115, f76, f91;
sub.f32 f116, f75, f92;
add.f32 f117, f76, f91;
add.f32 f118, f79, f104;
add.f32 f119, f80, f105;
sub.f32 f120, f79, f104;
sub.f32 f121, f80, f105;
add.f32 f122, %34, %56;
add.f32 f123, %36, %57;
sub.f32 f124, %34, %56;
sub.f32 f125, %36, %57;
add.f32 f126, %45, %66;
add.f32 f127, %47, %68;
sub.f32 f128, %45, %66;
sub.f32 f129, %47, %68;
add.f32 f130, f122, f126;
add.f32 f131, f123, f127;
sub.f32 f132, f122, f126;
sub.f32 f133, f123, f127;
add.f32 f134, f124, f129;
sub.f32 f135, f125, f128;
sub.f32 f136, f124, f129;
add.f32 f137, f125, f128;
add.f32 f138, %40, %61;
add.f32 f139, %41, %63;
sub.f32 f140, %40, %61;
sub.f32 f141, %41, %63;
add.f32 f142, %50, %72;
add.f32 f143, %52, %73;
sub.f32 f144, %50, %72;
sub.f32 f145, %52, %73;
add.f32 f146, f138, f142;
add.f32 f147, f139, f143;
sub.f32 f148, f138, f142;
sub.f32 f149, f139, f143;
add.f32 f150, f140, f145;
sub.f32 f151, f141, f144;
sub.f32 f152, f140, f145;
add.f32 f153, f141, f144;
mul.f32 f154, f150, 0f3F3504F3;
mul.f32 f155, f151, 0fBF3504F3;
sub.f32 f156, f154, f155;
mul.f32 f157, f151, 0f3F3504F3;
fma.rn.f32 f158, f150, 0fBF3504F3, f157;
mul.f32 f159, f152, 0fBF3504F3;
mul.f32 f160, f153, 0fBF3504F3;
sub.f32 f161, f159, f160;
add.f32 f162, f159, f160;
add.f32 f163, f130, f146;
add.f32 f164, f131, f147;
sub.f32 f165, f130, f146;
sub.f32 f166, f131, f147;
add.f32 f167, f134, f156;
add.f32 f168, f135, f158;
sub.f32 f169, f134, f156;
sub.f32 f170, f135, f158;
add.f32 f171, f132, f149;
sub.f32 f172, f133, f148;
sub.f32 f173, f132, f149;
add.f32 f174, f133, f148;
add.f32 f175, f136, f161;
add.f32 f176, f137, f162;
sub.f32 f177, f136, f161;
sub.f32 f178, f137, f162;
mul.f32 f179, f167, 0f3F6C835E;
mul.f32 f180, f168, 0fBEC3EF15;
sub.f32 f181, f179, f180;
mul.f32 f182, f168, 0f3F6C835E;
fma.rn.f32 f183, f167, 0fBEC3EF15, f182;
mul.f32 f184, f171, 0f3F3504F3;
mul.f32 f185, f172, 0fBF3504F3;
sub.f32 f186, f184, f185;
mul.f32 f187, f172, 0f3F3504F3;
fma.rn.f32 f188, f171, 0fBF3504F3, f187;
mul.f32 f189, f175, 0f3EC3EF15;
mul.f32 f190, f176, 0fBF6C835E;
sub.f32 f191, f189, f190;
mul.f32 f192, f176, 0f3EC3EF15;
fma.rn.f32 f193, f175, 0fBF6C835E, f192;
mul.f32 f194, f169, 0fBEC3EF15;
mul.f32 f195, f170, 0fBF6C835E;
sub.f32 f196, f194, f195;
mul.f32 f197, f170, 0fBEC3EF15;
fma.rn.f32 f198, f169, 0fBF6C835E, f197;
mul.f32 f199, f173, 0fBF3504F3;
mul.f32 f200, f174, 0fBF3504F3;
sub.f32 f201, f199, f200;
add.f32 f202, f199, f200;
mul.f32 f203, f177, 0fBF6C835E;
mul.f32 f204, f178, 0fBEC3EF15;
sub.f32 f205, f203, f204;
mul.f32 f206, f178, 0fBF6C835E;
fma.rn.f32 f207, f177, 0fBEC3EF15, f206;
add.f32 %1, f107, f164;
add.f32 %0, f106, f163;
add.f32 %3, f111, f183;
add.f32 %2, f110, f181;
add.f32 %5, f115, f188;
add.f32 %4, f114, f186;
add.f32 %7, f119, f193;
add.f32 %6, f118, f191;
sub.f32 %9, f109, f165;
add.f32 %8, f108, f166;
add.f32 %11, f113, f198;
add.f32 %10, f112, f196;
add.f32 %13, f117, f202;
add.f32 %12, f116, f201;
add.f32 %15, f121, f207;
add.f32 %14, f120, f205;
sub.f32 %17, f107, f164;
sub.f32 %16, f106, f163;
sub.f32 %19, f111, f183;
sub.f32 %18, f110, f181;
sub.f32 %21, f115, f188;
sub.f32 %20, f114, f186;
sub.f32 %23, f119, f193;
sub.f32 %22, f118, f191;
add.f32 %25, f109, f165;
sub.f32 %24, f108, f166;
sub.f32 %27, f113, f198;
sub.f32 %26, f112, f196;
sub.f32 %29, f117, f202;
sub.f32 %28, f116, f201;
sub.f32 %31, f121, f207;
sub.f32 %30, f120, f205;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y), "=f"(rmem[7].x), "=f"(rmem[7].y), "=f"(rmem[8].x), "=f"(rmem[8].y), "=f"(rmem[9].x), "=f"(rmem[9].y), "=f"(rmem[10].x), "=f"(rmem[10].y), "=f"(rmem[11].x), "=f"(rmem[11].y), "=f"(rmem[12].x), "=f"(rmem[12].y), "=f"(rmem[13].x), "=f"(rmem[13].y), "=f"(rmem[14].x), "=f"(rmem[14].y), "=f"(rmem[15].x), "=f"(rmem[15].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y), "f"(rmem[7].x), "f"(rmem[7].y), "f"(rmem[7].y), "f"(rmem[8].x), "f"(rmem[8].y), "f"(rmem[8].y), "f"(rmem[9].x), "f"(rmem[9].y), "f"(rmem[10].x), "f"(rmem[10].y), "f"(rmem[10].y), "f"(rmem[11].x), "f"(rmem[11].y), "f"(rmem[11].y), "f"(rmem[12].x), "f"(rmem[12].y), "f"(rmem[13].x), "f"(rmem[13].y), "f"(rmem[13].y), "f"(rmem[14].x), "f"(rmem[14].y), "f"(rmem[14].y), "f"(rmem[15].x), "f"(rmem[15].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<26, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<86>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.z;
shl.b32 r2, r1, 6;
mov.u32 r3, %8;
add.s32 r4, r3, r2;
add.f32 f17, %10, %15;
add.f32 f18, %11, %17;
sub.f32 f19, %10, %15;
sub.f32 f20, %11, %17;
add.f32 f21, %12, %18;
add.f32 f22, %14, %19;
sub.f32 f23, %12, %18;
sub.f32 f24, %14, %19;
add.f32 f25, f17, f21;
add.f32 f26, f18, f22;
sub.f32 f27, f17, f21;
sub.f32 f28, f18, f22;
add.f32 f29, f19, f24;
sub.f32 f30, f20, f23;
sub.f32 f31, f19, f24;
add.f32 f32, f20, f23;
mov.u32 r5, %tid.x;
and.b32 r6, r5, 3;
shl.b32 r7, r5, 3;
cvt.u64.u32 rd2, r7;
and.b64 rd3, rd2, 24;
mov.u64 rd4, %9;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f33, f34}, [rd5];
mul.f32 f37, f33, f29;
mul.f32 f38, f34, f30;
sub.f32 f39, f37, f38;
mul.f32 f40, f33, f30;
fma.rn.f32 f41, f34, f29, f40;
mul.f32 f42, f33, f33;
mul.f32 f43, f34, f34;
sub.f32 f44, f42, f43;
mul.f32 f45, f34, f33;
fma.rn.f32 f46, f34, f33, f45;
mul.f32 f47, f44, f27;
mul.f32 f48, f46, f28;
sub.f32 f49, f47, f48;
mul.f32 f50, f44, f28;
fma.rn.f32 f51, f46, f27, f50;
mul.f32 f52, f33, f44;
mul.f32 f53, f34, f46;
sub.f32 f54, f52, f53;
mul.f32 f55, f33, f46;
fma.rn.f32 f56, f34, f44, f55;
mul.f32 f57, f54, f31;
mul.f32 f58, f56, f32;
sub.f32 f59, f57, f58;
mul.f32 f60, f54, f32;
fma.rn.f32 f61, f56, f31, f60;
shl.b32 r8, r5, 4;
and.b32 r9, r8, -64;
add.s32 r10, r4, r9;
barrier.sync 0;
and.b32 r11, r8, 48;
add.s32 r12, r10, r11;
st.shared.v4.f32 [r12], {f25, f39, f49, f59};
barrier.sync 0;
mad.lo.s32 r13, r6, -12, r12;
ld.shared.f32 f62, [r13];
ld.shared.f32 f63, [r13+16];
ld.shared.f32 f64, [r13+32];
ld.shared.f32 f65, [r13+48];
barrier.sync 0;
st.shared.v4.f32 [r12], {f26, f41, f51, f61};
barrier.sync 0;
ld.shared.f32 f66, [r13];
ld.shared.f32 f67, [r13+16];
ld.shared.f32 f68, [r13+32];
ld.shared.f32 f69, [r13+48];
add.f32 f70, f62, f64;
add.f32 f71, f66, f68;
sub.f32 f72, f62, f64;
sub.f32 f73, f66, f68;
add.f32 f74, f63, f65;
add.f32 f75, f67, f69;
sub.f32 f76, f63, f65;
sub.f32 f77, f67, f69;
add.f32 %0, f70, f74;
add.f32 %1, f71, f75;
sub.f32 %3, f73, f76;
add.f32 %2, f72, f77;
sub.f32 %4, f70, f74;
sub.f32 %5, f71, f75;
add.f32 %7, f73, f76;
sub.f32 %6, f72, f77;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y): "r"(smem), "l"(lut_sp_4_16), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<27, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<94>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.z;
shl.b32 r2, r1, 7;
mov.u32 r3, %8;
add.s32 r4, r3, r2;
add.f32 f17, %10, %15;
add.f32 f18, %11, %17;
sub.f32 f19, %10, %15;
sub.f32 f20, %11, %17;
add.f32 f21, %12, %18;
add.f32 f22, %14, %19;
sub.f32 f23, %12, %18;
sub.f32 f24, %14, %19;
sub.f32 f25, f17, f21;
sub.f32 f26, f18, f22;
add.f32 f27, f19, f24;
sub.f32 f28, f20, f23;
sub.f32 f29, f19, f24;
add.f32 f30, f20, f23;
mov.u32 r5, %tid.x;
and.b32 r6, r5, 3;
shl.b32 r7, r5, 5;
and.b32 r8, r7, -128;
add.s32 r9, r4, r8;
shl.b32 r10, r5, 3;
cvt.u64.u32 rd2, r10;
and.b64 rd3, rd2, 24;
mov.u64 rd4, %9;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f31, f32}, [rd5];
mul.f32 f35, f31, f27;
mul.f32 f36, f32, f28;
mul.f32 f37, f31, f28;
mul.f32 f38, f31, f31;
mul.f32 f39, f32, f32;
sub.f32 f40, f38, f39;
mul.f32 f41, f32, f31;
fma.rn.f32 f42, f32, f31, f41;
mul.f32 f43, f40, f25;
mul.f32 f44, f42, f26;
mul.f32 f45, f40, f26;
mul.f32 f46, f31, f40;
mul.f32 f47, f32, f42;
sub.f32 f48, f46, f47;
mul.f32 f49, f31, f42;
fma.rn.f32 f50, f32, f40, f49;
mul.f32 f51, f48, f29;
mul.f32 f52, f50, f30;
mul.f32 f53, f48, f30;
barrier.sync 0;
and.b32 r11, r7, 96;
add.s32 r12, r9, r11;
add.f32 f54, f18, f22;
add.f32 f55, f17, f21;
fma.rn.f32 f56, f32, f27, f37;
sub.f32 f57, f35, f36;
st.shared.v4.f32 [r12], {f55, f54, f57, f56};
sub.f32 f58, f43, f44;
fma.rn.f32 f59, f42, f25, f45;
fma.rn.f32 f60, f50, f29, f53;
sub.f32 f61, f51, f52;
st.shared.v4.f32 [r12+16], {f58, f59, f61, f60};
barrier.sync 0;
mad.lo.s32 r13, r6, -24, r12;
ld.shared.v2.f32 {f62, f63}, [r13];
ld.shared.v2.f32 {f66, f67}, [r13+32];
ld.shared.v2.f32 {f70, f71}, [r13+64];
ld.shared.v2.f32 {f74, f75}, [r13+96];
add.f32 f78, f62, f70;
add.f32 f79, f63, f71;
sub.f32 f80, f62, f70;
sub.f32 f81, f63, f71;
add.f32 f82, f66, f74;
add.f32 f83, f67, f75;
sub.f32 f84, f66, f74;
sub.f32 f85, f67, f75;
add.f32 %1, f79, f83;
add.f32 %0, f78, f82;
sub.f32 %3, f81, f84;
add.f32 %2, f80, f85;
sub.f32 %5, f79, f83;
sub.f32 %4, f78, f82;
add.f32 %7, f81, f84;
sub.f32 %6, f80, f85;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y): "r"(smem), "l"(lut_sp_4_16), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<28, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<191>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.z;
shl.b32 r2, r1, 6;
mov.u32 r3, %16;
add.s32 r4, r3, r2;
add.f32 f33, %18, %28;
add.f32 f34, %19, %30;
sub.f32 f35, %18, %28;
sub.f32 f36, %19, %30;
add.f32 f37, %23, %34;
add.f32 f38, %25, %35;
sub.f32 f39, %23, %34;
sub.f32 f40, %25, %35;
add.f32 f41, f33, f37;
add.f32 f42, f34, f38;
sub.f32 f43, f33, f37;
sub.f32 f44, f34, f38;
add.f32 f45, f35, f40;
sub.f32 f46, f36, f39;
sub.f32 f47, f35, f40;
add.f32 f48, f36, f39;
add.f32 f49, %20, %31;
add.f32 f50, %22, %33;
sub.f32 f51, %20, %31;
sub.f32 f52, %22, %33;
add.f32 f53, %26, %36;
add.f32 f54, %27, %37;
sub.f32 f55, %26, %36;
sub.f32 f56, %27, %37;
add.f32 f57, f49, f53;
add.f32 f58, f50, f54;
sub.f32 f59, f49, f53;
sub.f32 f60, f50, f54;
add.f32 f61, f51, f56;
sub.f32 f62, f52, f55;
sub.f32 f63, f51, f56;
add.f32 f64, f52, f55;
mul.f32 f65, f61, 0f3F3504F3;
mul.f32 f66, f62, 0fBF3504F3;
sub.f32 f67, f65, f66;
mul.f32 f68, f62, 0f3F3504F3;
fma.rn.f32 f69, f61, 0fBF3504F3, f68;
mul.f32 f70, f63, 0fBF3504F3;
mul.f32 f71, f64, 0fBF3504F3;
sub.f32 f72, f70, f71;
add.f32 f73, f70, f71;
add.f32 f74, f41, f57;
add.f32 f75, f42, f58;
sub.f32 f76, f41, f57;
sub.f32 f77, f42, f58;
add.f32 f78, f45, f67;
add.f32 f79, f46, f69;
sub.f32 f80, f45, f67;
sub.f32 f81, f46, f69;
add.f32 f82, f43, f60;
sub.f32 f83, f44, f59;
sub.f32 f84, f43, f60;
add.f32 f85, f44, f59;
add.f32 f86, f47, f72;
add.f32 f87, f48, f73;
sub.f32 f88, f47, f72;
sub.f32 f89, f48, f73;
mov.u32 r5, %tid.x;
and.b32 r6, r5, 1;
shl.b32 r7, r5, 3;
cvt.u64.u32 rd2, r7;
and.b64 rd3, rd2, 8;
mov.u64 rd4, %17;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f90, f91}, [rd5];
mul.f32 f94, f90, f78;
mul.f32 f95, f91, f79;
sub.f32 f96, f94, f95;
mul.f32 f97, f90, f79;
fma.rn.f32 f98, f91, f78, f97;
mul.f32 f99, f90, f90;
mul.f32 f100, f91, f91;
sub.f32 f101, f99, f100;
mul.f32 f102, f91, f90;
fma.rn.f32 f103, f91, f90, f102;
mul.f32 f104, f101, f82;
mul.f32 f105, f103, f83;
sub.f32 f106, f104, f105;
mul.f32 f107, f101, f83;
fma.rn.f32 f108, f103, f82, f107;
mul.f32 f109, f90, f101;
mul.f32 f110, f91, f103;
sub.f32 f111, f109, f110;
mul.f32 f112, f90, f103;
fma.rn.f32 f113, f91, f101, f112;
mul.f32 f114, f111, f86;
mul.f32 f115, f113, f87;
sub.f32 f116, f114, f115;
mul.f32 f117, f111, f87;
fma.rn.f32 f118, f113, f86, f117;
mul.f32 f119, f90, f111;
mul.f32 f120, f91, f113;
sub.f32 f121, f119, f120;
mul.f32 f122, f90, f113;
fma.rn.f32 f123, f91, f111, f122;
mul.f32 f124, f121, f76;
mul.f32 f125, f123, f77;
sub.f32 f126, f124, f125;
mul.f32 f127, f121, f77;
fma.rn.f32 f128, f123, f76, f127;
mul.f32 f129, f90, f121;
mul.f32 f130, f91, f123;
sub.f32 f131, f129, f130;
mul.f32 f132, f90, f123;
fma.rn.f32 f133, f91, f121, f132;
mul.f32 f134, f131, f80;
mul.f32 f135, f133, f81;
sub.f32 f136, f134, f135;
mul.f32 f137, f131, f81;
fma.rn.f32 f138, f133, f80, f137;
mul.f32 f139, f90, f131;
mul.f32 f140, f91, f133;
sub.f32 f141, f139, f140;
mul.f32 f142, f90, f133;
fma.rn.f32 f143, f91, f131, f142;
mul.f32 f144, f141, f84;
mul.f32 f145, f143, f85;
sub.f32 f146, f144, f145;
mul.f32 f147, f141, f85;
fma.rn.f32 f148, f143, f84, f147;
mul.f32 f149, f90, f141;
mul.f32 f150, f91, f143;
sub.f32 f151, f149, f150;
mul.f32 f152, f90, f143;
fma.rn.f32 f153, f91, f141, f152;
mul.f32 f154, f151, f88;
mul.f32 f155, f153, f89;
sub.f32 f156, f154, f155;
mul.f32 f157, f151, f89;
fma.rn.f32 f158, f153, f88, f157;
shl.b32 r8, r5, 5;
and.b32 r9, r8, -64;
add.s32 r10, r4, r9;
barrier.sync 0;
and.b32 r11, r8, 32;
add.s32 r12, r10, r11;
st.shared.v4.f32 [r12], {f74, f96, f106, f116};
st.shared.v4.f32 [r12+16], {f126, f136, f146, f156};
barrier.sync 0;
mad.lo.s32 r13, r6, -28, r12;
ld.shared.f32 f159, [r13];
ld.shared.f32 f160, [r13+8];
ld.shared.f32 f161, [r13+16];
ld.shared.f32 f162, [r13+24];
ld.shared.f32 f163, [r13+32];
ld.shared.f32 f164, [r13+40];
ld.shared.f32 f165, [r13+48];
ld.shared.f32 f166, [r13+56];
barrier.sync 0;
st.shared.v4.f32 [r12], {f75, f98, f108, f118};
st.shared.v4.f32 [r12+16], {f128, f138, f148, f158};
barrier.sync 0;
ld.shared.f32 f167, [r13];
ld.shared.f32 f168, [r13+8];
ld.shared.f32 f169, [r13+16];
ld.shared.f32 f170, [r13+24];
ld.shared.f32 f171, [r13+32];
ld.shared.f32 f172, [r13+40];
ld.shared.f32 f173, [r13+48];
ld.shared.f32 f174, [r13+56];
add.f32 %0, f159, f163;
add.f32 %1, f167, f171;
add.f32 %2, f160, f164;
add.f32 %3, f168, f172;
add.f32 %4, f161, f165;
add.f32 %5, f169, f173;
add.f32 %6, f162, f166;
add.f32 %7, f170, f174;
sub.f32 %8, f159, f163;
sub.f32 %9, f167, f171;
sub.f32 %10, f160, f164;
sub.f32 %11, f168, f172;
sub.f32 %12, f161, f165;
sub.f32 %13, f169, f173;
sub.f32 %14, f162, f166;
sub.f32 %15, f170, f174;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y), "=f"(rmem[7].x), "=f"(rmem[7].y): "r"(smem), "l"(lut_sp_8_16), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y), "f"(rmem[7].x), "f"(rmem[7].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<29, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<207>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.z;
shl.b32 r2, r1, 7;
mov.u32 r3, %16;
add.s32 r4, r3, r2;
add.f32 f33, %18, %28;
add.f32 f34, %19, %30;
sub.f32 f35, %18, %28;
sub.f32 f36, %19, %30;
add.f32 f37, %23, %34;
add.f32 f38, %25, %35;
sub.f32 f39, %23, %34;
sub.f32 f40, %25, %35;
add.f32 f41, f33, f37;
add.f32 f42, f34, f38;
sub.f32 f43, f33, f37;
sub.f32 f44, f34, f38;
add.f32 f45, f35, f40;
sub.f32 f46, f36, f39;
sub.f32 f47, f35, f40;
add.f32 f48, f36, f39;
add.f32 f49, %20, %31;
add.f32 f50, %22, %33;
sub.f32 f51, %20, %31;
sub.f32 f52, %22, %33;
add.f32 f53, %26, %36;
add.f32 f54, %27, %37;
sub.f32 f55, %26, %36;
sub.f32 f56, %27, %37;
add.f32 f57, f49, f53;
add.f32 f58, f50, f54;
sub.f32 f59, f49, f53;
sub.f32 f60, f50, f54;
add.f32 f61, f51, f56;
sub.f32 f62, f52, f55;
sub.f32 f63, f51, f56;
add.f32 f64, f52, f55;
mul.f32 f65, f61, 0f3F3504F3;
mul.f32 f66, f62, 0fBF3504F3;
sub.f32 f67, f65, f66;
mul.f32 f68, f62, 0f3F3504F3;
fma.rn.f32 f69, f61, 0fBF3504F3, f68;
mul.f32 f70, f63, 0fBF3504F3;
mul.f32 f71, f64, 0fBF3504F3;
sub.f32 f72, f70, f71;
add.f32 f73, f70, f71;
sub.f32 f74, f41, f57;
sub.f32 f75, f42, f58;
add.f32 f76, f45, f67;
add.f32 f77, f46, f69;
sub.f32 f78, f45, f67;
sub.f32 f79, f46, f69;
add.f32 f80, f43, f60;
sub.f32 f81, f44, f59;
sub.f32 f82, f43, f60;
add.f32 f83, f44, f59;
add.f32 f84, f47, f72;
add.f32 f85, f48, f73;
sub.f32 f86, f47, f72;
sub.f32 f87, f48, f73;
mov.u32 r5, %tid.x;
and.b32 r6, r5, 1;
shl.b32 r7, r5, 6;
and.b32 r8, r7, -128;
add.s32 r9, r4, r8;
shl.b32 r10, r5, 3;
cvt.u64.u32 rd2, r10;
and.b64 rd3, rd2, 8;
mov.u64 rd4, %17;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f88, f89}, [rd5];
mul.f32 f92, f88, f76;
mul.f32 f93, f89, f77;
mul.f32 f94, f88, f77;
mul.f32 f95, f88, f88;
mul.f32 f96, f89, f89;
sub.f32 f97, f95, f96;
mul.f32 f98, f89, f88;
fma.rn.f32 f99, f89, f88, f98;
mul.f32 f100, f97, f80;
mul.f32 f101, f99, f81;
mul.f32 f102, f97, f81;
mul.f32 f103, f88, f97;
mul.f32 f104, f89, f99;
sub.f32 f105, f103, f104;
mul.f32 f106, f88, f99;
fma.rn.f32 f107, f89, f97, f106;
mul.f32 f108, f105, f84;
mul.f32 f109, f107, f85;
mul.f32 f110, f105, f85;
mul.f32 f111, f88, f105;
mul.f32 f112, f89, f107;
sub.f32 f113, f111, f112;
mul.f32 f114, f88, f107;
fma.rn.f32 f115, f89, f105, f114;
mul.f32 f116, f113, f74;
mul.f32 f117, f115, f75;
mul.f32 f118, f113, f75;
mul.f32 f119, f88, f113;
mul.f32 f120, f89, f115;
sub.f32 f121, f119, f120;
mul.f32 f122, f88, f115;
fma.rn.f32 f123, f89, f113, f122;
mul.f32 f124, f121, f78;
mul.f32 f125, f123, f79;
mul.f32 f126, f121, f79;
mul.f32 f127, f88, f121;
mul.f32 f128, f89, f123;
sub.f32 f129, f127, f128;
mul.f32 f130, f88, f123;
fma.rn.f32 f131, f89, f121, f130;
mul.f32 f132, f129, f82;
mul.f32 f133, f131, f83;
mul.f32 f134, f129, f83;
mul.f32 f135, f88, f129;
mul.f32 f136, f89, f131;
sub.f32 f137, f135, f136;
mul.f32 f138, f88, f131;
fma.rn.f32 f139, f89, f129, f138;
mul.f32 f140, f137, f86;
mul.f32 f141, f139, f87;
mul.f32 f142, f137, f87;
barrier.sync 0;
and.b32 r11, r7, 64;
add.s32 r12, r9, r11;
add.f32 f143, f42, f58;
add.f32 f144, f41, f57;
fma.rn.f32 f145, f89, f76, f94;
sub.f32 f146, f92, f93;
st.shared.v4.f32 [r12], {f144, f143, f146, f145};
fma.rn.f32 f147, f99, f80, f102;
sub.f32 f148, f100, f101;
sub.f32 f149, f108, f109;
fma.rn.f32 f150, f107, f84, f110;
st.shared.v4.f32 [r12+16], {f148, f147, f149, f150};
fma.rn.f32 f151, f115, f74, f118;
sub.f32 f152, f116, f117;
fma.rn.f32 f153, f123, f78, f126;
sub.f32 f154, f124, f125;
st.shared.v4.f32 [r12+32], {f152, f151, f154, f153};
fma.rn.f32 f155, f131, f82, f134;
sub.f32 f156, f132, f133;
fma.rn.f32 f157, f139, f86, f142;
sub.f32 f158, f140, f141;
st.shared.v4.f32 [r12+48], {f156, f155, f158, f157};
barrier.sync 0;
mad.lo.s32 r13, r6, -56, r12;
ld.shared.v2.f32 {f159, f160}, [r13];
ld.shared.v2.f32 {f163, f164}, [r13+16];
ld.shared.v2.f32 {f167, f168}, [r13+32];
ld.shared.v2.f32 {f171, f172}, [r13+48];
ld.shared.v2.f32 {f175, f176}, [r13+64];
ld.shared.v2.f32 {f179, f180}, [r13+80];
ld.shared.v2.f32 {f183, f184}, [r13+96];
ld.shared.v2.f32 {f187, f188}, [r13+112];
add.f32 %1, f160, f176;
add.f32 %0, f159, f175;
add.f32 %3, f164, f180;
add.f32 %2, f163, f179;
add.f32 %5, f168, f184;
add.f32 %4, f167, f183;
add.f32 %7, f172, f188;
add.f32 %6, f171, f187;
sub.f32 %9, f160, f176;
sub.f32 %8, f159, f175;
sub.f32 %11, f164, f180;
sub.f32 %10, f163, f179;
sub.f32 %13, f168, f184;
sub.f32 %12, f167, f183;
sub.f32 %15, f172, f188;
sub.f32 %14, f171, f187;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y), "=f"(rmem[7].x), "=f"(rmem[7].y): "r"(smem), "l"(lut_sp_8_16), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y), "f"(rmem[7].x), "f"(rmem[7].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<30, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<64>;
.reg .b32 r<28>;
.reg .b64 rd<12>;
mov.u32 r1, %tid.z;
shl.b32 r2, r1, 6;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
add.f32 f9, %8, %10;
add.f32 f10, %9, %11;
sub.f32 f11, %8, %10;
sub.f32 f12, %9, %11;
mov.u32 r5, %tid.x;
shl.b32 r6, r5, 3;
cvt.u64.u32 rd2, r6;
and.b64 rd3, rd2, 56;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f13, f14}, [rd5];
mul.f32 f17, f13, f11;
mul.f32 f18, f14, f12;
sub.f32 f19, f17, f18;
mul.f32 f20, f13, f12;
fma.rn.f32 f21, f14, f11, f20;
and.b32 r7, r6, -64;
add.s32 r8, r4, r7;
barrier.sync 0;
and.b32 r9, r6, 56;
add.s32 r10, r8, r9;
st.shared.v2.f32 [r10], {f9, f19};
barrier.sync 0;
shl.b32 r11, r5, 2;
and.b32 r12, r11, 28;
sub.s32 r13, r10, r12;
ld.shared.f32 f22, [r13];
ld.shared.f32 f23, [r13+32];
barrier.sync 0;
st.shared.v2.f32 [r10], {f10, f21};
barrier.sync 0;
ld.shared.f32 f24, [r13];
ld.shared.f32 f25, [r13+32];
add.f32 f26, f22, f23;
add.f32 f27, f24, f25;
sub.f32 f28, f22, f23;
sub.f32 f29, f24, f25;
bfe.u32 r14, r5, 1, 2;
mul.wide.u32 rd6, r14, 8;
mov.u64 rd7, %6;
add.s64 rd8, rd7, rd6;
ld.global.v2.f32 {f30, f31}, [rd8];
mul.f32 f34, f30, f28;
mul.f32 f35, f31, f29;
sub.f32 f36, f34, f35;
mul.f32 f37, f30, f29;
fma.rn.f32 f38, f31, f28, f37;
and.b32 r15, r11, 4;
add.s32 r16, r8, r15;
barrier.sync 0;
and.b32 r17, r6, 48;
add.s32 r18, r16, r17;
st.shared.f32 [r18], f26;
st.shared.f32 [r18+8], f36;
barrier.sync 0;
and.b32 r19, r11, 24;
sub.s32 r20, r18, r19;
ld.shared.f32 f39, [r20];
ld.shared.f32 f40, [r20+32];
barrier.sync 0;
st.shared.f32 [r18], f27;
st.shared.f32 [r18+8], f38;
barrier.sync 0;
ld.shared.f32 f41, [r20];
ld.shared.f32 f42, [r20+32];
add.f32 f43, f39, f40;
add.f32 f44, f41, f42;
sub.f32 f45, f39, f40;
sub.f32 f46, f41, f42;
bfe.u32 r21, r5, 2, 1;
mul.wide.u32 rd9, r21, 8;
mov.u64 rd10, %7;
add.s64 rd11, rd10, rd9;
ld.global.v2.f32 {f47, f48}, [rd11];
mul.f32 f51, f47, f45;
mul.f32 f52, f48, f46;
sub.f32 f53, f51, f52;
mul.f32 f54, f47, f46;
fma.rn.f32 f55, f48, f45, f54;
and.b32 r22, r11, 12;
add.s32 r23, r8, r22;
barrier.sync 0;
and.b32 r24, r6, 32;
add.s32 r25, r23, r24;
st.shared.f32 [r25], f43;
st.shared.f32 [r25+16], f53;
barrier.sync 0;
and.b32 r26, r11, 16;
sub.s32 r27, r25, r26;
ld.shared.f32 f56, [r27];
ld.shared.f32 f57, [r27+32];
barrier.sync 0;
st.shared.f32 [r25], f44;
st.shared.f32 [r25+16], f55;
barrier.sync 0;
ld.shared.f32 f58, [r27];
ld.shared.f32 f59, [r27+32];
add.f32 %0, f56, f57;
add.f32 %1, f58, f59;
sub.f32 %2, f56, f57;
sub.f32 %3, f58, f59;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y): "r"(smem), "l"(lut_sp_2_16), "l"(lut_sp_2_8), "l"(lut_sp_2_4), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<31, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<76>;
.reg .b32 r<28>;
.reg .b64 rd<12>;
mov.u32 r1, %tid.z;
shl.b32 r2, r1, 7;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
sub.f32 f9, %8, %10;
sub.f32 f10, %9, %11;
mov.u32 r5, %tid.x;
shl.b32 r6, r5, 4;
and.b32 r7, r6, -128;
add.s32 r8, r4, r7;
shl.b32 r9, r5, 3;
cvt.u64.u32 rd2, r9;
and.b64 rd3, rd2, 56;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f11, f12}, [rd5];
mul.f32 f15, f11, f9;
mul.f32 f16, f12, f10;
mul.f32 f17, f11, f10;
barrier.sync 0;
and.b32 r10, r6, 112;
add.s32 r11, r8, r10;
add.f32 f18, %9, %11;
add.f32 f19, %8, %10;
st.shared.v2.f32 [r11], {f19, f18};
sub.f32 f20, f15, f16;
fma.rn.f32 f21, f12, f9, f17;
st.shared.v2.f32 [r11+8], {f20, f21};
barrier.sync 0;
and.b32 r12, r9, 56;
sub.s32 r13, r11, r12;
ld.shared.v2.f32 {f22, f23}, [r13];
ld.shared.v2.f32 {f26, f27}, [r13+64];
sub.f32 f30, f22, f26;
sub.f32 f31, f23, f27;
bfe.u32 r14, r5, 1, 2;
mul.wide.u32 rd6, r14, 8;
mov.u64 rd7, %6;
add.s64 rd8, rd7, rd6;
ld.global.v2.f32 {f32, f33}, [rd8];
mul.f32 f36, f32, f30;
mul.f32 f37, f33, f31;
mul.f32 f38, f32, f31;
and.b32 r15, r9, 8;
add.s32 r16, r8, r15;
barrier.sync 0;
and.b32 r17, r6, 96;
add.s32 r18, r16, r17;
add.f32 f39, f23, f27;
add.f32 f40, f22, f26;
st.shared.v2.f32 [r18], {f40, f39};
fma.rn.f32 f41, f33, f30, f38;
sub.f32 f42, f36, f37;
st.shared.v2.f32 [r18+16], {f42, f41};
barrier.sync 0;
and.b32 r19, r9, 48;
sub.s32 r20, r18, r19;
ld.shared.v2.f32 {f43, f44}, [r20];
ld.shared.v2.f32 {f47, f48}, [r20+64];
sub.f32 f51, f43, f47;
sub.f32 f52, f44, f48;
bfe.u32 r21, r5, 2, 1;
mul.wide.u32 rd9, r21, 8;
mov.u64 rd10, %7;
add.s64 rd11, rd10, rd9;
ld.global.v2.f32 {f53, f54}, [rd11];
mul.f32 f57, f53, f51;
mul.f32 f58, f54, f52;
mul.f32 f59, f53, f52;
and.b32 r22, r9, 24;
add.s32 r23, r8, r22;
barrier.sync 0;
and.b32 r24, r6, 64;
add.s32 r25, r23, r24;
add.f32 f60, f44, f48;
add.f32 f61, f43, f47;
st.shared.v2.f32 [r25], {f61, f60};
fma.rn.f32 f62, f54, f51, f59;
sub.f32 f63, f57, f58;
st.shared.v2.f32 [r25+32], {f63, f62};
barrier.sync 0;
and.b32 r26, r9, 32;
sub.s32 r27, r25, r26;
ld.shared.v2.f32 {f64, f65}, [r27];
ld.shared.v2.f32 {f68, f69}, [r27+64];
add.f32 %1, f65, f69;
add.f32 %0, f64, f68;
sub.f32 %3, f65, f69;
sub.f32 %2, f64, f68;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y): "r"(smem), "l"(lut_sp_2_16), "l"(lut_sp_2_8), "l"(lut_sp_2_4), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y));
};


#endif
