//Copyright (c) 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_121_FP32_INV_PTX_HPP
#define CUFFTDX_FFT_121_FP32_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<381, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<488>;
.reg .b32 r<11>;
.reg .b64 rd<7>;
mov.u32 r1, %tid.z;
add.f32 f45, %26, %50;
add.f32 f46, %28, %51;
sub.f32 f47, %26, %50;
sub.f32 f48, %28, %51;
add.f32 f49, %29, %48;
add.f32 f50, %31, %49;
sub.f32 f51, %29, %48;
sub.f32 f52, %31, %49;
add.f32 f53, %32, %45;
add.f32 f54, %33, %47;
sub.f32 f55, %32, %45;
sub.f32 f56, %33, %47;
add.f32 f57, %34, %42;
add.f32 f58, %36, %44;
sub.f32 f59, %34, %42;
sub.f32 f60, %36, %44;
add.f32 f61, %37, %40;
add.f32 f62, %39, %41;
sub.f32 f63, %37, %40;
sub.f32 f64, %39, %41;
mov.u32 r2, %22;
mad.lo.s32 r3, r1, 968, r2;
add.f32 f65, %24, f45;
add.f32 f66, %25, f46;
add.f32 f67, f65, f49;
add.f32 f68, f66, f50;
add.f32 f69, f67, f53;
add.f32 f70, f68, f54;
add.f32 f71, f69, f57;
add.f32 f72, f70, f58;
fma.rn.f32 f73, f45, 0f3F575C64, %24;
fma.rn.f32 f74, f48, 0f3F0A6770, 0f00000000;
fma.rn.f32 f75, f46, 0f3F575C64, %25;
fma.rn.f32 f76, f47, 0f3F0A6770, 0f00000000;
fma.rn.f32 f77, f49, 0f3ED4B147, f73;
fma.rn.f32 f78, f52, 0f3F68DDA4, f74;
fma.rn.f32 f79, f50, 0f3ED4B147, f75;
fma.rn.f32 f80, f51, 0f3F68DDA4, f76;
fma.rn.f32 f81, f53, 0fBE11BAFB, f77;
fma.rn.f32 f82, f56, 0f3F7D64F0, f78;
fma.rn.f32 f83, f54, 0fBE11BAFB, f79;
fma.rn.f32 f84, f55, 0f3F7D64F0, f80;
fma.rn.f32 f85, f57, 0fBF27A4F4, f81;
fma.rn.f32 f86, f60, 0f3F4178CE, f82;
fma.rn.f32 f87, f58, 0fBF27A4F4, f83;
fma.rn.f32 f88, f59, 0f3F4178CE, f84;
fma.rn.f32 f89, f61, 0fBF75A155, f85;
fma.rn.f32 f90, f64, 0f3E903F40, f86;
fma.rn.f32 f91, f62, 0fBF75A155, f87;
fma.rn.f32 f92, f63, 0f3E903F40, f88;
sub.f32 f93, f89, f90;
add.f32 f94, f92, f91;
add.f32 f95, f90, f89;
sub.f32 f96, f91, f92;
fma.rn.f32 f97, f45, 0f3ED4B147, %24;
fma.rn.f32 f98, f48, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f99, f46, 0f3ED4B147, %25;
fma.rn.f32 f100, f47, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f101, f49, 0fBF27A4F4, f97;
fma.rn.f32 f102, f52, 0f3F4178CE, f98;
fma.rn.f32 f103, f50, 0fBF27A4F4, f99;
fma.rn.f32 f104, f51, 0f3F4178CE, f100;
fma.rn.f32 f105, f53, 0fBF75A155, f101;
fma.rn.f32 f106, f56, 0fBE903F40, f102;
fma.rn.f32 f107, f54, 0fBF75A155, f103;
fma.rn.f32 f108, f55, 0fBE903F40, f104;
fma.rn.f32 f109, f57, 0fBE11BAFB, f105;
fma.rn.f32 f110, f60, 0fBF7D64F0, f106;
fma.rn.f32 f111, f58, 0fBE11BAFB, f107;
fma.rn.f32 f112, f59, 0fBF7D64F0, f108;
fma.rn.f32 f113, f61, 0f3F575C64, f109;
fma.rn.f32 f114, f64, 0fBF0A6770, f110;
fma.rn.f32 f115, f62, 0f3F575C64, f111;
fma.rn.f32 f116, f63, 0fBF0A6770, f112;
sub.f32 f117, f113, f114;
add.f32 f118, f116, f115;
add.f32 f119, f114, f113;
sub.f32 f120, f115, f116;
fma.rn.f32 f121, f45, 0fBE11BAFB, %24;
fma.rn.f32 f122, f48, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f123, f46, 0fBE11BAFB, %25;
fma.rn.f32 f124, f47, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f125, f49, 0fBF75A155, f121;
fma.rn.f32 f126, f52, 0fBE903F40, f122;
fma.rn.f32 f127, f50, 0fBF75A155, f123;
fma.rn.f32 f128, f51, 0fBE903F40, f124;
fma.rn.f32 f129, f53, 0f3ED4B147, f125;
fma.rn.f32 f130, f56, 0fBF68DDA4, f126;
fma.rn.f32 f131, f54, 0f3ED4B147, f127;
fma.rn.f32 f132, f55, 0fBF68DDA4, f128;
fma.rn.f32 f133, f57, 0f3F575C64, f129;
fma.rn.f32 f134, f60, 0f3F0A6770, f130;
fma.rn.f32 f135, f58, 0f3F575C64, f131;
fma.rn.f32 f136, f59, 0f3F0A6770, f132;
fma.rn.f32 f137, f61, 0fBF27A4F4, f133;
fma.rn.f32 f138, f64, 0f3F4178CE, f134;
fma.rn.f32 f139, f62, 0fBF27A4F4, f135;
fma.rn.f32 f140, f63, 0f3F4178CE, f136;
sub.f32 f141, f137, f138;
add.f32 f142, f140, f139;
add.f32 f143, f138, f137;
sub.f32 f144, f139, f140;
fma.rn.f32 f145, f45, 0fBF27A4F4, %24;
fma.rn.f32 f146, f48, 0f3F4178CE, 0f00000000;
fma.rn.f32 f147, f46, 0fBF27A4F4, %25;
fma.rn.f32 f148, f47, 0f3F4178CE, 0f00000000;
fma.rn.f32 f149, f49, 0fBE11BAFB, f145;
fma.rn.f32 f150, f52, 0fBF7D64F0, f146;
fma.rn.f32 f151, f50, 0fBE11BAFB, f147;
fma.rn.f32 f152, f51, 0fBF7D64F0, f148;
fma.rn.f32 f153, f53, 0f3F575C64, f149;
fma.rn.f32 f154, f56, 0f3F0A6770, f150;
fma.rn.f32 f155, f54, 0f3F575C64, f151;
fma.rn.f32 f156, f55, 0f3F0A6770, f152;
fma.rn.f32 f157, f57, 0fBF75A155, f153;
fma.rn.f32 f158, f60, 0f3E903F40, f154;
fma.rn.f32 f159, f58, 0fBF75A155, f155;
fma.rn.f32 f160, f59, 0f3E903F40, f156;
fma.rn.f32 f161, f61, 0f3ED4B147, f157;
fma.rn.f32 f162, f64, 0fBF68DDA4, f158;
fma.rn.f32 f163, f62, 0f3ED4B147, f159;
fma.rn.f32 f164, f63, 0fBF68DDA4, f160;
sub.f32 f165, f161, f162;
add.f32 f166, f164, f163;
add.f32 f167, f162, f161;
sub.f32 f168, f163, f164;
fma.rn.f32 f169, f45, 0fBF75A155, %24;
fma.rn.f32 f170, f48, 0f3E903F40, 0f00000000;
fma.rn.f32 f171, f46, 0fBF75A155, %25;
fma.rn.f32 f172, f47, 0f3E903F40, 0f00000000;
fma.rn.f32 f173, f49, 0f3F575C64, f169;
fma.rn.f32 f174, f52, 0fBF0A6770, f170;
fma.rn.f32 f175, f50, 0f3F575C64, f171;
fma.rn.f32 f176, f51, 0fBF0A6770, f172;
fma.rn.f32 f177, f53, 0fBF27A4F4, f173;
fma.rn.f32 f178, f56, 0f3F4178CE, f174;
fma.rn.f32 f179, f54, 0fBF27A4F4, f175;
fma.rn.f32 f180, f55, 0f3F4178CE, f176;
fma.rn.f32 f181, f57, 0f3ED4B147, f177;
fma.rn.f32 f182, f60, 0fBF68DDA4, f178;
fma.rn.f32 f183, f58, 0f3ED4B147, f179;
fma.rn.f32 f184, f59, 0fBF68DDA4, f180;
fma.rn.f32 f185, f61, 0fBE11BAFB, f181;
fma.rn.f32 f186, f64, 0f3F7D64F0, f182;
fma.rn.f32 f187, f62, 0fBE11BAFB, f183;
fma.rn.f32 f188, f63, 0f3F7D64F0, f184;
sub.f32 f189, f185, f186;
add.f32 f190, f188, f187;
add.f32 f191, f186, f185;
sub.f32 f192, f187, f188;
mov.u32 r4, %tid.x;
mul.wide.u32 rd2, r4, -1171354717;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r5, rd3;
mul.lo.s32 r6, r5, 11;
sub.s32 r7, r4, r6;
mad.lo.s32 r8, r5, 968, r3;
mul.wide.u32 rd4, r7, 8;
mov.u64 rd5, %23;
add.s64 rd6, rd5, rd4;
ld.global.v2.f32 {f193, f194}, [rd6];
mul.f32 f197, f94, f194;
mul.f32 f198, f93, f194;
mul.f32 f199, f193, f94;
mul.f32 f200, f193, f193;
mul.f32 f201, f194, f194;
sub.f32 f202, f200, f201;
mul.f32 f203, f194, f193;
fma.rn.f32 f204, f194, f193, f203;
mul.f32 f205, f118, f204;
mul.f32 f206, f117, f204;
mul.f32 f207, f202, f118;
mul.f32 f208, f193, f202;
mul.f32 f209, f194, f204;
sub.f32 f210, f208, f209;
mul.f32 f211, f193, f204;
fma.rn.f32 f212, f194, f202, f211;
mul.f32 f213, f142, f212;
mul.f32 f214, f141, f212;
mul.f32 f215, f210, f142;
mul.f32 f216, f193, f210;
mul.f32 f217, f194, f212;
sub.f32 f218, f216, f217;
mul.f32 f219, f193, f212;
fma.rn.f32 f220, f194, f210, f219;
mul.f32 f221, f166, f220;
mul.f32 f222, f165, f220;
mul.f32 f223, f218, f166;
mul.f32 f224, f193, f218;
mul.f32 f225, f194, f220;
sub.f32 f226, f224, f225;
mul.f32 f227, f193, f220;
fma.rn.f32 f228, f194, f218, f227;
mul.f32 f229, f190, f228;
mul.f32 f230, f189, f228;
mul.f32 f231, f226, f190;
mul.f32 f232, f193, f226;
mul.f32 f233, f194, f228;
sub.f32 f234, f232, f233;
mul.f32 f235, f193, f228;
fma.rn.f32 f236, f194, f226, f235;
mul.f32 f237, f192, f236;
mul.f32 f238, f191, f236;
mul.f32 f239, f234, f192;
mul.f32 f240, f193, f234;
mul.f32 f241, f194, f236;
sub.f32 f242, f240, f241;
mul.f32 f243, f193, f236;
fma.rn.f32 f244, f194, f234, f243;
mul.f32 f245, f168, f244;
mul.f32 f246, f167, f244;
mul.f32 f247, f242, f168;
mul.f32 f248, f193, f242;
mul.f32 f249, f194, f244;
sub.f32 f250, f248, f249;
mul.f32 f251, f193, f244;
fma.rn.f32 f252, f194, f242, f251;
mul.f32 f253, f144, f252;
mul.f32 f254, f143, f252;
mul.f32 f255, f250, f144;
mul.f32 f256, f193, f250;
mul.f32 f257, f194, f252;
sub.f32 f258, f256, f257;
mul.f32 f259, f193, f252;
fma.rn.f32 f260, f194, f250, f259;
mul.f32 f261, f120, f260;
mul.f32 f262, f119, f260;
mul.f32 f263, f258, f120;
mul.f32 f264, f193, f258;
mul.f32 f265, f194, f260;
sub.f32 f266, f264, f265;
mul.f32 f267, f193, f260;
fma.rn.f32 f268, f194, f258, f267;
mul.f32 f269, f96, f268;
mul.f32 f270, f95, f268;
mul.f32 f271, f266, f96;
barrier.sync 0;
mad.lo.s32 r9, r7, 88, r8;
add.f32 f272, f72, f62;
add.f32 f273, f71, f61;
st.shared.v2.f32 [r9], {f273, f272};
fma.rn.f32 f274, f193, f93, f197;
sub.f32 f275, f199, f198;
st.shared.v2.f32 [r9+8], {f274, f275};
fma.rn.f32 f276, f202, f117, f205;
sub.f32 f277, f207, f206;
st.shared.v2.f32 [r9+16], {f276, f277};
sub.f32 f278, f215, f214;
fma.rn.f32 f279, f210, f141, f213;
st.shared.v2.f32 [r9+24], {f279, f278};
fma.rn.f32 f280, f218, f165, f221;
sub.f32 f281, f223, f222;
st.shared.v2.f32 [r9+32], {f280, f281};
fma.rn.f32 f282, f226, f189, f229;
sub.f32 f283, f231, f230;
st.shared.v2.f32 [r9+40], {f282, f283};
fma.rn.f32 f284, f234, f191, f237;
sub.f32 f285, f239, f238;
st.shared.v2.f32 [r9+48], {f284, f285};
fma.rn.f32 f286, f242, f167, f245;
sub.f32 f287, f247, f246;
st.shared.v2.f32 [r9+56], {f286, f287};
fma.rn.f32 f288, f250, f143, f253;
sub.f32 f289, f255, f254;
st.shared.v2.f32 [r9+64], {f288, f289};
fma.rn.f32 f290, f258, f119, f261;
sub.f32 f291, f263, f262;
st.shared.v2.f32 [r9+72], {f290, f291};
fma.rn.f32 f292, f266, f95, f269;
sub.f32 f293, f271, f270;
st.shared.v2.f32 [r9+80], {f292, f293};
barrier.sync 0;
mad.lo.s32 r10, r7, -80, r9;
ld.shared.v2.f32 {f294, f295}, [r10];
ld.shared.v2.f32 {f298, f299}, [r10+88];
ld.shared.v2.f32 {f302, f303}, [r10+176];
ld.shared.v2.f32 {f306, f307}, [r10+264];
ld.shared.v2.f32 {f310, f311}, [r10+352];
ld.shared.v2.f32 {f314, f315}, [r10+440];
ld.shared.v2.f32 {f318, f319}, [r10+528];
ld.shared.v2.f32 {f322, f323}, [r10+616];
ld.shared.v2.f32 {f326, f327}, [r10+704];
ld.shared.v2.f32 {f330, f331}, [r10+792];
ld.shared.v2.f32 {f334, f335}, [r10+880];
add.f32 f338, f298, f334;
add.f32 f339, f299, f335;
sub.f32 f340, f298, f334;
sub.f32 f341, f299, f335;
add.f32 f342, f302, f330;
add.f32 f343, f303, f331;
sub.f32 f344, f302, f330;
sub.f32 f345, f303, f331;
add.f32 f346, f306, f326;
add.f32 f347, f307, f327;
sub.f32 f348, f306, f326;
sub.f32 f349, f307, f327;
add.f32 f350, f310, f322;
add.f32 f351, f311, f323;
sub.f32 f352, f310, f322;
sub.f32 f353, f311, f323;
add.f32 f354, f314, f318;
add.f32 f355, f315, f319;
sub.f32 f356, f314, f318;
sub.f32 f357, f315, f319;
add.f32 f358, f294, f338;
add.f32 f359, f295, f339;
add.f32 f360, f358, f342;
add.f32 f361, f359, f343;
add.f32 f362, f360, f346;
add.f32 f363, f361, f347;
add.f32 f364, f362, f350;
add.f32 f365, f363, f351;
fma.rn.f32 f366, f338, 0f3F575C64, f294;
fma.rn.f32 f367, f341, 0f3F0A6770, 0f00000000;
fma.rn.f32 f368, f339, 0f3F575C64, f295;
fma.rn.f32 f369, f340, 0f3F0A6770, 0f00000000;
fma.rn.f32 f370, f342, 0f3ED4B147, f366;
fma.rn.f32 f371, f345, 0f3F68DDA4, f367;
fma.rn.f32 f372, f343, 0f3ED4B147, f368;
fma.rn.f32 f373, f344, 0f3F68DDA4, f369;
fma.rn.f32 f374, f346, 0fBE11BAFB, f370;
fma.rn.f32 f375, f349, 0f3F7D64F0, f371;
fma.rn.f32 f376, f347, 0fBE11BAFB, f372;
fma.rn.f32 f377, f348, 0f3F7D64F0, f373;
fma.rn.f32 f378, f350, 0fBF27A4F4, f374;
fma.rn.f32 f379, f353, 0f3F4178CE, f375;
fma.rn.f32 f380, f351, 0fBF27A4F4, f376;
fma.rn.f32 f381, f352, 0f3F4178CE, f377;
fma.rn.f32 f382, f354, 0fBF75A155, f378;
fma.rn.f32 f383, f357, 0f3E903F40, f379;
fma.rn.f32 f384, f355, 0fBF75A155, f380;
fma.rn.f32 f385, f356, 0f3E903F40, f381;
fma.rn.f32 f386, f338, 0f3ED4B147, f294;
fma.rn.f32 f387, f341, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f388, f339, 0f3ED4B147, f295;
fma.rn.f32 f389, f340, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f390, f342, 0fBF27A4F4, f386;
fma.rn.f32 f391, f345, 0f3F4178CE, f387;
fma.rn.f32 f392, f343, 0fBF27A4F4, f388;
fma.rn.f32 f393, f344, 0f3F4178CE, f389;
fma.rn.f32 f394, f346, 0fBF75A155, f390;
fma.rn.f32 f395, f349, 0fBE903F40, f391;
fma.rn.f32 f396, f347, 0fBF75A155, f392;
fma.rn.f32 f397, f348, 0fBE903F40, f393;
fma.rn.f32 f398, f350, 0fBE11BAFB, f394;
fma.rn.f32 f399, f353, 0fBF7D64F0, f395;
fma.rn.f32 f400, f351, 0fBE11BAFB, f396;
fma.rn.f32 f401, f352, 0fBF7D64F0, f397;
fma.rn.f32 f402, f354, 0f3F575C64, f398;
fma.rn.f32 f403, f357, 0fBF0A6770, f399;
fma.rn.f32 f404, f355, 0f3F575C64, f400;
fma.rn.f32 f405, f356, 0fBF0A6770, f401;
fma.rn.f32 f406, f338, 0fBE11BAFB, f294;
fma.rn.f32 f407, f341, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f408, f339, 0fBE11BAFB, f295;
fma.rn.f32 f409, f340, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f410, f342, 0fBF75A155, f406;
fma.rn.f32 f411, f345, 0fBE903F40, f407;
fma.rn.f32 f412, f343, 0fBF75A155, f408;
fma.rn.f32 f413, f344, 0fBE903F40, f409;
fma.rn.f32 f414, f346, 0f3ED4B147, f410;
fma.rn.f32 f415, f349, 0fBF68DDA4, f411;
fma.rn.f32 f416, f347, 0f3ED4B147, f412;
fma.rn.f32 f417, f348, 0fBF68DDA4, f413;
fma.rn.f32 f418, f350, 0f3F575C64, f414;
fma.rn.f32 f419, f353, 0f3F0A6770, f415;
fma.rn.f32 f420, f351, 0f3F575C64, f416;
fma.rn.f32 f421, f352, 0f3F0A6770, f417;
fma.rn.f32 f422, f354, 0fBF27A4F4, f418;
fma.rn.f32 f423, f357, 0f3F4178CE, f419;
fma.rn.f32 f424, f355, 0fBF27A4F4, f420;
fma.rn.f32 f425, f356, 0f3F4178CE, f421;
fma.rn.f32 f426, f338, 0fBF27A4F4, f294;
fma.rn.f32 f427, f341, 0f3F4178CE, 0f00000000;
fma.rn.f32 f428, f339, 0fBF27A4F4, f295;
fma.rn.f32 f429, f340, 0f3F4178CE, 0f00000000;
fma.rn.f32 f430, f342, 0fBE11BAFB, f426;
fma.rn.f32 f431, f345, 0fBF7D64F0, f427;
fma.rn.f32 f432, f343, 0fBE11BAFB, f428;
fma.rn.f32 f433, f344, 0fBF7D64F0, f429;
fma.rn.f32 f434, f346, 0f3F575C64, f430;
fma.rn.f32 f435, f349, 0f3F0A6770, f431;
fma.rn.f32 f436, f347, 0f3F575C64, f432;
fma.rn.f32 f437, f348, 0f3F0A6770, f433;
fma.rn.f32 f438, f350, 0fBF75A155, f434;
fma.rn.f32 f439, f353, 0f3E903F40, f435;
fma.rn.f32 f440, f351, 0fBF75A155, f436;
fma.rn.f32 f441, f352, 0f3E903F40, f437;
fma.rn.f32 f442, f354, 0f3ED4B147, f438;
fma.rn.f32 f443, f357, 0fBF68DDA4, f439;
fma.rn.f32 f444, f355, 0f3ED4B147, f440;
fma.rn.f32 f445, f356, 0fBF68DDA4, f441;
fma.rn.f32 f446, f338, 0fBF75A155, f294;
fma.rn.f32 f447, f341, 0f3E903F40, 0f00000000;
fma.rn.f32 f448, f339, 0fBF75A155, f295;
fma.rn.f32 f449, f340, 0f3E903F40, 0f00000000;
fma.rn.f32 f450, f342, 0f3F575C64, f446;
fma.rn.f32 f451, f345, 0fBF0A6770, f447;
fma.rn.f32 f452, f343, 0f3F575C64, f448;
fma.rn.f32 f453, f344, 0fBF0A6770, f449;
fma.rn.f32 f454, f346, 0fBF27A4F4, f450;
fma.rn.f32 f455, f349, 0f3F4178CE, f451;
fma.rn.f32 f456, f347, 0fBF27A4F4, f452;
fma.rn.f32 f457, f348, 0f3F4178CE, f453;
fma.rn.f32 f458, f350, 0f3ED4B147, f454;
fma.rn.f32 f459, f353, 0fBF68DDA4, f455;
fma.rn.f32 f460, f351, 0f3ED4B147, f456;
fma.rn.f32 f461, f352, 0fBF68DDA4, f457;
fma.rn.f32 f462, f354, 0fBE11BAFB, f458;
fma.rn.f32 f463, f357, 0f3F7D64F0, f459;
fma.rn.f32 f464, f355, 0fBE11BAFB, f460;
fma.rn.f32 f465, f356, 0f3F7D64F0, f461;
add.f32 %1, f365, f355;
add.f32 %0, f364, f354;
add.f32 %3, f385, f384;
sub.f32 %2, f382, f383;
add.f32 %5, f405, f404;
sub.f32 %4, f402, f403;
add.f32 %7, f425, f424;
sub.f32 %6, f422, f423;
add.f32 %9, f445, f444;
sub.f32 %8, f442, f443;
add.f32 %11, f465, f464;
sub.f32 %10, f462, f463;
sub.f32 %13, f464, f465;
add.f32 %12, f463, f462;
sub.f32 %15, f444, f445;
add.f32 %14, f443, f442;
sub.f32 %17, f424, f425;
add.f32 %16, f423, f422;
sub.f32 %19, f404, f405;
add.f32 %18, f403, f402;
sub.f32 %21, f384, f385;
add.f32 %20, f383, f382;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y), "=f"(rmem[7].x), "=f"(rmem[7].y), "=f"(rmem[8].x), "=f"(rmem[8].y), "=f"(rmem[9].x), "=f"(rmem[9].y), "=f"(rmem[10].x), "=f"(rmem[10].y): "r"(smem), "l"(lut_sp_11_121), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y), "f"(rmem[7].x), "f"(rmem[7].y), "f"(rmem[7].y), "f"(rmem[8].x), "f"(rmem[8].y), "f"(rmem[8].y), "f"(rmem[9].x), "f"(rmem[9].y), "f"(rmem[10].x), "f"(rmem[10].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<382, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<466>;
.reg .b32 r<11>;
.reg .b64 rd<7>;
mov.u32 r1, %tid.z;
add.f32 f45, %26, %50;
add.f32 f46, %28, %51;
sub.f32 f47, %26, %50;
sub.f32 f48, %28, %51;
add.f32 f49, %29, %48;
add.f32 f50, %31, %49;
sub.f32 f51, %29, %48;
sub.f32 f52, %31, %49;
add.f32 f53, %32, %45;
add.f32 f54, %33, %47;
sub.f32 f55, %32, %45;
sub.f32 f56, %33, %47;
add.f32 f57, %34, %42;
add.f32 f58, %36, %44;
sub.f32 f59, %34, %42;
sub.f32 f60, %36, %44;
add.f32 f61, %37, %40;
add.f32 f62, %39, %41;
sub.f32 f63, %37, %40;
sub.f32 f64, %39, %41;
mov.u32 r2, %22;
mad.lo.s32 r3, r1, 484, r2;
add.f32 f65, %24, f45;
add.f32 f66, %25, f46;
add.f32 f67, f65, f49;
add.f32 f68, f66, f50;
add.f32 f69, f67, f53;
add.f32 f70, f68, f54;
add.f32 f71, f69, f57;
add.f32 f72, f70, f58;
add.f32 f73, f71, f61;
add.f32 f74, f72, f62;
fma.rn.f32 f75, f45, 0f3F575C64, %24;
fma.rn.f32 f76, f48, 0f3F0A6770, 0f00000000;
fma.rn.f32 f77, f46, 0f3F575C64, %25;
fma.rn.f32 f78, f47, 0f3F0A6770, 0f00000000;
fma.rn.f32 f79, f49, 0f3ED4B147, f75;
fma.rn.f32 f80, f52, 0f3F68DDA4, f76;
fma.rn.f32 f81, f50, 0f3ED4B147, f77;
fma.rn.f32 f82, f51, 0f3F68DDA4, f78;
fma.rn.f32 f83, f53, 0fBE11BAFB, f79;
fma.rn.f32 f84, f56, 0f3F7D64F0, f80;
fma.rn.f32 f85, f54, 0fBE11BAFB, f81;
fma.rn.f32 f86, f55, 0f3F7D64F0, f82;
fma.rn.f32 f87, f57, 0fBF27A4F4, f83;
fma.rn.f32 f88, f60, 0f3F4178CE, f84;
fma.rn.f32 f89, f58, 0fBF27A4F4, f85;
fma.rn.f32 f90, f59, 0f3F4178CE, f86;
fma.rn.f32 f91, f61, 0fBF75A155, f87;
fma.rn.f32 f92, f64, 0f3E903F40, f88;
fma.rn.f32 f93, f62, 0fBF75A155, f89;
fma.rn.f32 f94, f63, 0f3E903F40, f90;
sub.f32 f95, f91, f92;
add.f32 f96, f94, f93;
add.f32 f97, f92, f91;
sub.f32 f98, f93, f94;
fma.rn.f32 f99, f45, 0f3ED4B147, %24;
fma.rn.f32 f100, f48, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f101, f46, 0f3ED4B147, %25;
fma.rn.f32 f102, f47, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f103, f49, 0fBF27A4F4, f99;
fma.rn.f32 f104, f52, 0f3F4178CE, f100;
fma.rn.f32 f105, f50, 0fBF27A4F4, f101;
fma.rn.f32 f106, f51, 0f3F4178CE, f102;
fma.rn.f32 f107, f53, 0fBF75A155, f103;
fma.rn.f32 f108, f56, 0fBE903F40, f104;
fma.rn.f32 f109, f54, 0fBF75A155, f105;
fma.rn.f32 f110, f55, 0fBE903F40, f106;
fma.rn.f32 f111, f57, 0fBE11BAFB, f107;
fma.rn.f32 f112, f60, 0fBF7D64F0, f108;
fma.rn.f32 f113, f58, 0fBE11BAFB, f109;
fma.rn.f32 f114, f59, 0fBF7D64F0, f110;
fma.rn.f32 f115, f61, 0f3F575C64, f111;
fma.rn.f32 f116, f64, 0fBF0A6770, f112;
fma.rn.f32 f117, f62, 0f3F575C64, f113;
fma.rn.f32 f118, f63, 0fBF0A6770, f114;
sub.f32 f119, f115, f116;
add.f32 f120, f118, f117;
add.f32 f121, f116, f115;
sub.f32 f122, f117, f118;
fma.rn.f32 f123, f45, 0fBE11BAFB, %24;
fma.rn.f32 f124, f48, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f125, f46, 0fBE11BAFB, %25;
fma.rn.f32 f126, f47, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f127, f49, 0fBF75A155, f123;
fma.rn.f32 f128, f52, 0fBE903F40, f124;
fma.rn.f32 f129, f50, 0fBF75A155, f125;
fma.rn.f32 f130, f51, 0fBE903F40, f126;
fma.rn.f32 f131, f53, 0f3ED4B147, f127;
fma.rn.f32 f132, f56, 0fBF68DDA4, f128;
fma.rn.f32 f133, f54, 0f3ED4B147, f129;
fma.rn.f32 f134, f55, 0fBF68DDA4, f130;
fma.rn.f32 f135, f57, 0f3F575C64, f131;
fma.rn.f32 f136, f60, 0f3F0A6770, f132;
fma.rn.f32 f137, f58, 0f3F575C64, f133;
fma.rn.f32 f138, f59, 0f3F0A6770, f134;
fma.rn.f32 f139, f61, 0fBF27A4F4, f135;
fma.rn.f32 f140, f64, 0f3F4178CE, f136;
fma.rn.f32 f141, f62, 0fBF27A4F4, f137;
fma.rn.f32 f142, f63, 0f3F4178CE, f138;
sub.f32 f143, f139, f140;
add.f32 f144, f142, f141;
add.f32 f145, f140, f139;
sub.f32 f146, f141, f142;
fma.rn.f32 f147, f45, 0fBF27A4F4, %24;
fma.rn.f32 f148, f48, 0f3F4178CE, 0f00000000;
fma.rn.f32 f149, f46, 0fBF27A4F4, %25;
fma.rn.f32 f150, f47, 0f3F4178CE, 0f00000000;
fma.rn.f32 f151, f49, 0fBE11BAFB, f147;
fma.rn.f32 f152, f52, 0fBF7D64F0, f148;
fma.rn.f32 f153, f50, 0fBE11BAFB, f149;
fma.rn.f32 f154, f51, 0fBF7D64F0, f150;
fma.rn.f32 f155, f53, 0f3F575C64, f151;
fma.rn.f32 f156, f56, 0f3F0A6770, f152;
fma.rn.f32 f157, f54, 0f3F575C64, f153;
fma.rn.f32 f158, f55, 0f3F0A6770, f154;
fma.rn.f32 f159, f57, 0fBF75A155, f155;
fma.rn.f32 f160, f60, 0f3E903F40, f156;
fma.rn.f32 f161, f58, 0fBF75A155, f157;
fma.rn.f32 f162, f59, 0f3E903F40, f158;
fma.rn.f32 f163, f61, 0f3ED4B147, f159;
fma.rn.f32 f164, f64, 0fBF68DDA4, f160;
fma.rn.f32 f165, f62, 0f3ED4B147, f161;
fma.rn.f32 f166, f63, 0fBF68DDA4, f162;
sub.f32 f167, f163, f164;
add.f32 f168, f166, f165;
add.f32 f169, f164, f163;
sub.f32 f170, f165, f166;
fma.rn.f32 f171, f45, 0fBF75A155, %24;
fma.rn.f32 f172, f48, 0f3E903F40, 0f00000000;
fma.rn.f32 f173, f46, 0fBF75A155, %25;
fma.rn.f32 f174, f47, 0f3E903F40, 0f00000000;
fma.rn.f32 f175, f49, 0f3F575C64, f171;
fma.rn.f32 f176, f52, 0fBF0A6770, f172;
fma.rn.f32 f177, f50, 0f3F575C64, f173;
fma.rn.f32 f178, f51, 0fBF0A6770, f174;
fma.rn.f32 f179, f53, 0fBF27A4F4, f175;
fma.rn.f32 f180, f56, 0f3F4178CE, f176;
fma.rn.f32 f181, f54, 0fBF27A4F4, f177;
fma.rn.f32 f182, f55, 0f3F4178CE, f178;
fma.rn.f32 f183, f57, 0f3ED4B147, f179;
fma.rn.f32 f184, f60, 0fBF68DDA4, f180;
fma.rn.f32 f185, f58, 0f3ED4B147, f181;
fma.rn.f32 f186, f59, 0fBF68DDA4, f182;
fma.rn.f32 f187, f61, 0fBE11BAFB, f183;
fma.rn.f32 f188, f64, 0f3F7D64F0, f184;
fma.rn.f32 f189, f62, 0fBE11BAFB, f185;
fma.rn.f32 f190, f63, 0f3F7D64F0, f186;
sub.f32 f191, f187, f188;
add.f32 f192, f190, f189;
add.f32 f193, f188, f187;
sub.f32 f194, f189, f190;
mov.u32 r4, %tid.x;
mul.wide.u32 rd2, r4, -1171354717;
shr.u64 rd3, rd2, 35;
cvt.u32.u64 r5, rd3;
mul.lo.s32 r6, r5, 11;
sub.s32 r7, r4, r6;
mul.wide.u32 rd4, r7, 8;
mov.u64 rd5, %23;
add.s64 rd6, rd5, rd4;
ld.global.v2.f32 {f195, f196}, [rd6];
mul.f32 f199, f96, f196;
fma.rn.f32 f200, f195, f95, f199;
mul.f32 f201, f95, f196;
mul.f32 f202, f195, f96;
sub.f32 f203, f202, f201;
mul.f32 f204, f195, f195;
mul.f32 f205, f196, f196;
sub.f32 f206, f204, f205;
mul.f32 f207, f196, f195;
fma.rn.f32 f208, f196, f195, f207;
mul.f32 f209, f120, f208;
fma.rn.f32 f210, f206, f119, f209;
mul.f32 f211, f119, f208;
mul.f32 f212, f206, f120;
sub.f32 f213, f212, f211;
mul.f32 f214, f195, f206;
mul.f32 f215, f196, f208;
sub.f32 f216, f214, f215;
mul.f32 f217, f195, f208;
fma.rn.f32 f218, f196, f206, f217;
mul.f32 f219, f144, f218;
fma.rn.f32 f220, f216, f143, f219;
mul.f32 f221, f143, f218;
mul.f32 f222, f216, f144;
sub.f32 f223, f222, f221;
mul.f32 f224, f195, f216;
mul.f32 f225, f196, f218;
sub.f32 f226, f224, f225;
mul.f32 f227, f195, f218;
fma.rn.f32 f228, f196, f216, f227;
mul.f32 f229, f168, f228;
fma.rn.f32 f230, f226, f167, f229;
mul.f32 f231, f167, f228;
mul.f32 f232, f226, f168;
sub.f32 f233, f232, f231;
mul.f32 f234, f195, f226;
mul.f32 f235, f196, f228;
sub.f32 f236, f234, f235;
mul.f32 f237, f195, f228;
fma.rn.f32 f238, f196, f226, f237;
mul.f32 f239, f192, f238;
fma.rn.f32 f240, f236, f191, f239;
mul.f32 f241, f191, f238;
mul.f32 f242, f236, f192;
sub.f32 f243, f242, f241;
mul.f32 f244, f195, f236;
mul.f32 f245, f196, f238;
sub.f32 f246, f244, f245;
mul.f32 f247, f195, f238;
fma.rn.f32 f248, f196, f236, f247;
mul.f32 f249, f194, f248;
fma.rn.f32 f250, f246, f193, f249;
mul.f32 f251, f193, f248;
mul.f32 f252, f246, f194;
sub.f32 f253, f252, f251;
mul.f32 f254, f195, f246;
mul.f32 f255, f196, f248;
sub.f32 f256, f254, f255;
mul.f32 f257, f195, f248;
fma.rn.f32 f258, f196, f246, f257;
mul.f32 f259, f170, f258;
fma.rn.f32 f260, f256, f169, f259;
mul.f32 f261, f169, f258;
mul.f32 f262, f256, f170;
sub.f32 f263, f262, f261;
mul.f32 f264, f195, f256;
mul.f32 f265, f196, f258;
sub.f32 f266, f264, f265;
mul.f32 f267, f195, f258;
fma.rn.f32 f268, f196, f256, f267;
mul.f32 f269, f146, f268;
fma.rn.f32 f270, f266, f145, f269;
mul.f32 f271, f145, f268;
mul.f32 f272, f266, f146;
sub.f32 f273, f272, f271;
mul.f32 f274, f195, f266;
mul.f32 f275, f196, f268;
sub.f32 f276, f274, f275;
mul.f32 f277, f195, f268;
fma.rn.f32 f278, f196, f266, f277;
mul.f32 f279, f122, f278;
fma.rn.f32 f280, f276, f121, f279;
mul.f32 f281, f121, f278;
mul.f32 f282, f276, f122;
sub.f32 f283, f282, f281;
mul.f32 f284, f195, f276;
mul.f32 f285, f196, f278;
sub.f32 f286, f284, f285;
mul.f32 f287, f195, f278;
fma.rn.f32 f288, f196, f276, f287;
mul.f32 f289, f98, f288;
fma.rn.f32 f290, f286, f97, f289;
mul.f32 f291, f97, f288;
mul.f32 f292, f286, f98;
sub.f32 f293, f292, f291;
mad.lo.s32 r8, r5, 484, r3;
barrier.sync 0;
mad.lo.s32 r9, r7, 44, r8;
st.shared.f32 [r9], f73;
st.shared.f32 [r9+4], f200;
st.shared.f32 [r9+8], f210;
st.shared.f32 [r9+12], f220;
st.shared.f32 [r9+16], f230;
st.shared.f32 [r9+20], f240;
st.shared.f32 [r9+24], f250;
st.shared.f32 [r9+28], f260;
st.shared.f32 [r9+32], f270;
st.shared.f32 [r9+36], f280;
st.shared.f32 [r9+40], f290;
barrier.sync 0;
mad.lo.s32 r10, r7, -40, r9;
ld.shared.f32 f294, [r10];
ld.shared.f32 f295, [r10+44];
ld.shared.f32 f296, [r10+88];
ld.shared.f32 f297, [r10+132];
ld.shared.f32 f298, [r10+176];
ld.shared.f32 f299, [r10+220];
ld.shared.f32 f300, [r10+264];
ld.shared.f32 f301, [r10+308];
ld.shared.f32 f302, [r10+352];
ld.shared.f32 f303, [r10+396];
ld.shared.f32 f304, [r10+440];
barrier.sync 0;
st.shared.f32 [r9], f74;
st.shared.f32 [r9+4], f203;
st.shared.f32 [r9+8], f213;
st.shared.f32 [r9+12], f223;
st.shared.f32 [r9+16], f233;
st.shared.f32 [r9+20], f243;
st.shared.f32 [r9+24], f253;
st.shared.f32 [r9+28], f263;
st.shared.f32 [r9+32], f273;
st.shared.f32 [r9+36], f283;
st.shared.f32 [r9+40], f293;
barrier.sync 0;
ld.shared.f32 f305, [r10];
ld.shared.f32 f306, [r10+44];
ld.shared.f32 f307, [r10+88];
ld.shared.f32 f308, [r10+132];
ld.shared.f32 f309, [r10+176];
ld.shared.f32 f310, [r10+220];
ld.shared.f32 f311, [r10+264];
ld.shared.f32 f312, [r10+308];
ld.shared.f32 f313, [r10+352];
ld.shared.f32 f314, [r10+396];
ld.shared.f32 f315, [r10+440];
add.f32 f316, f295, f304;
add.f32 f317, f306, f315;
sub.f32 f318, f295, f304;
sub.f32 f319, f306, f315;
add.f32 f320, f296, f303;
add.f32 f321, f307, f314;
sub.f32 f322, f296, f303;
sub.f32 f323, f307, f314;
add.f32 f324, f297, f302;
add.f32 f325, f308, f313;
sub.f32 f326, f297, f302;
sub.f32 f327, f308, f313;
add.f32 f328, f298, f301;
add.f32 f329, f309, f312;
sub.f32 f330, f298, f301;
sub.f32 f331, f309, f312;
add.f32 f332, f299, f300;
add.f32 f333, f310, f311;
sub.f32 f334, f299, f300;
sub.f32 f335, f310, f311;
add.f32 f336, f294, f316;
add.f32 f337, f305, f317;
add.f32 f338, f336, f320;
add.f32 f339, f337, f321;
add.f32 f340, f338, f324;
add.f32 f341, f339, f325;
add.f32 f342, f340, f328;
add.f32 f343, f341, f329;
fma.rn.f32 f344, f316, 0f3F575C64, f294;
fma.rn.f32 f345, f319, 0f3F0A6770, 0f00000000;
fma.rn.f32 f346, f317, 0f3F575C64, f305;
fma.rn.f32 f347, f318, 0f3F0A6770, 0f00000000;
fma.rn.f32 f348, f320, 0f3ED4B147, f344;
fma.rn.f32 f349, f323, 0f3F68DDA4, f345;
fma.rn.f32 f350, f321, 0f3ED4B147, f346;
fma.rn.f32 f351, f322, 0f3F68DDA4, f347;
fma.rn.f32 f352, f324, 0fBE11BAFB, f348;
fma.rn.f32 f353, f327, 0f3F7D64F0, f349;
fma.rn.f32 f354, f325, 0fBE11BAFB, f350;
fma.rn.f32 f355, f326, 0f3F7D64F0, f351;
fma.rn.f32 f356, f328, 0fBF27A4F4, f352;
fma.rn.f32 f357, f331, 0f3F4178CE, f353;
fma.rn.f32 f358, f329, 0fBF27A4F4, f354;
fma.rn.f32 f359, f330, 0f3F4178CE, f355;
fma.rn.f32 f360, f332, 0fBF75A155, f356;
fma.rn.f32 f361, f335, 0f3E903F40, f357;
fma.rn.f32 f362, f333, 0fBF75A155, f358;
fma.rn.f32 f363, f334, 0f3E903F40, f359;
fma.rn.f32 f364, f316, 0f3ED4B147, f294;
fma.rn.f32 f365, f319, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f366, f317, 0f3ED4B147, f305;
fma.rn.f32 f367, f318, 0f3F68DDA4, 0f00000000;
fma.rn.f32 f368, f320, 0fBF27A4F4, f364;
fma.rn.f32 f369, f323, 0f3F4178CE, f365;
fma.rn.f32 f370, f321, 0fBF27A4F4, f366;
fma.rn.f32 f371, f322, 0f3F4178CE, f367;
fma.rn.f32 f372, f324, 0fBF75A155, f368;
fma.rn.f32 f373, f327, 0fBE903F40, f369;
fma.rn.f32 f374, f325, 0fBF75A155, f370;
fma.rn.f32 f375, f326, 0fBE903F40, f371;
fma.rn.f32 f376, f328, 0fBE11BAFB, f372;
fma.rn.f32 f377, f331, 0fBF7D64F0, f373;
fma.rn.f32 f378, f329, 0fBE11BAFB, f374;
fma.rn.f32 f379, f330, 0fBF7D64F0, f375;
fma.rn.f32 f380, f332, 0f3F575C64, f376;
fma.rn.f32 f381, f335, 0fBF0A6770, f377;
fma.rn.f32 f382, f333, 0f3F575C64, f378;
fma.rn.f32 f383, f334, 0fBF0A6770, f379;
fma.rn.f32 f384, f316, 0fBE11BAFB, f294;
fma.rn.f32 f385, f319, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f386, f317, 0fBE11BAFB, f305;
fma.rn.f32 f387, f318, 0f3F7D64F0, 0f00000000;
fma.rn.f32 f388, f320, 0fBF75A155, f384;
fma.rn.f32 f389, f323, 0fBE903F40, f385;
fma.rn.f32 f390, f321, 0fBF75A155, f386;
fma.rn.f32 f391, f322, 0fBE903F40, f387;
fma.rn.f32 f392, f324, 0f3ED4B147, f388;
fma.rn.f32 f393, f327, 0fBF68DDA4, f389;
fma.rn.f32 f394, f325, 0f3ED4B147, f390;
fma.rn.f32 f395, f326, 0fBF68DDA4, f391;
fma.rn.f32 f396, f328, 0f3F575C64, f392;
fma.rn.f32 f397, f331, 0f3F0A6770, f393;
fma.rn.f32 f398, f329, 0f3F575C64, f394;
fma.rn.f32 f399, f330, 0f3F0A6770, f395;
fma.rn.f32 f400, f332, 0fBF27A4F4, f396;
fma.rn.f32 f401, f335, 0f3F4178CE, f397;
fma.rn.f32 f402, f333, 0fBF27A4F4, f398;
fma.rn.f32 f403, f334, 0f3F4178CE, f399;
fma.rn.f32 f404, f316, 0fBF27A4F4, f294;
fma.rn.f32 f405, f319, 0f3F4178CE, 0f00000000;
fma.rn.f32 f406, f317, 0fBF27A4F4, f305;
fma.rn.f32 f407, f318, 0f3F4178CE, 0f00000000;
fma.rn.f32 f408, f320, 0fBE11BAFB, f404;
fma.rn.f32 f409, f323, 0fBF7D64F0, f405;
fma.rn.f32 f410, f321, 0fBE11BAFB, f406;
fma.rn.f32 f411, f322, 0fBF7D64F0, f407;
fma.rn.f32 f412, f324, 0f3F575C64, f408;
fma.rn.f32 f413, f327, 0f3F0A6770, f409;
fma.rn.f32 f414, f325, 0f3F575C64, f410;
fma.rn.f32 f415, f326, 0f3F0A6770, f411;
fma.rn.f32 f416, f328, 0fBF75A155, f412;
fma.rn.f32 f417, f331, 0f3E903F40, f413;
fma.rn.f32 f418, f329, 0fBF75A155, f414;
fma.rn.f32 f419, f330, 0f3E903F40, f415;
fma.rn.f32 f420, f332, 0f3ED4B147, f416;
fma.rn.f32 f421, f335, 0fBF68DDA4, f417;
fma.rn.f32 f422, f333, 0f3ED4B147, f418;
fma.rn.f32 f423, f334, 0fBF68DDA4, f419;
fma.rn.f32 f424, f316, 0fBF75A155, f294;
fma.rn.f32 f425, f319, 0f3E903F40, 0f00000000;
fma.rn.f32 f426, f317, 0fBF75A155, f305;
fma.rn.f32 f427, f318, 0f3E903F40, 0f00000000;
fma.rn.f32 f428, f320, 0f3F575C64, f424;
fma.rn.f32 f429, f323, 0fBF0A6770, f425;
fma.rn.f32 f430, f321, 0f3F575C64, f426;
fma.rn.f32 f431, f322, 0fBF0A6770, f427;
fma.rn.f32 f432, f324, 0fBF27A4F4, f428;
fma.rn.f32 f433, f327, 0f3F4178CE, f429;
fma.rn.f32 f434, f325, 0fBF27A4F4, f430;
fma.rn.f32 f435, f326, 0f3F4178CE, f431;
fma.rn.f32 f436, f328, 0f3ED4B147, f432;
fma.rn.f32 f437, f331, 0fBF68DDA4, f433;
fma.rn.f32 f438, f329, 0f3ED4B147, f434;
fma.rn.f32 f439, f330, 0fBF68DDA4, f435;
fma.rn.f32 f440, f332, 0fBE11BAFB, f436;
fma.rn.f32 f441, f335, 0f3F7D64F0, f437;
fma.rn.f32 f442, f333, 0fBE11BAFB, f438;
fma.rn.f32 f443, f334, 0f3F7D64F0, f439;
add.f32 %0, f342, f332;
add.f32 %1, f343, f333;
add.f32 %3, f363, f362;
sub.f32 %2, f360, f361;
add.f32 %5, f383, f382;
sub.f32 %4, f380, f381;
add.f32 %7, f403, f402;
sub.f32 %6, f400, f401;
add.f32 %9, f423, f422;
sub.f32 %8, f420, f421;
add.f32 %11, f443, f442;
sub.f32 %10, f440, f441;
sub.f32 %13, f442, f443;
add.f32 %12, f441, f440;
sub.f32 %15, f422, f423;
add.f32 %14, f421, f420;
sub.f32 %17, f402, f403;
add.f32 %16, f401, f400;
sub.f32 %19, f382, f383;
add.f32 %18, f381, f380;
sub.f32 %21, f362, f363;
add.f32 %20, f361, f360;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y), "=f"(rmem[7].x), "=f"(rmem[7].y), "=f"(rmem[8].x), "=f"(rmem[8].y), "=f"(rmem[9].x), "=f"(rmem[9].y), "=f"(rmem[10].x), "=f"(rmem[10].y): "r"(smem), "l"(lut_sp_11_121), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y), "f"(rmem[7].x), "f"(rmem[7].y), "f"(rmem[7].y), "f"(rmem[8].x), "f"(rmem[8].y), "f"(rmem[8].y), "f"(rmem[9].x), "f"(rmem[9].y), "f"(rmem[10].x), "f"(rmem[10].y));
};


#endif
