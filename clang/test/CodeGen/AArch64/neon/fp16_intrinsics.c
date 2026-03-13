#include <arm_fp16.h>
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// LLVM-LABEL: @test_vaddh_f16(
// CIR-LABEL: @test_vaddh_f16(
float16_t test_vaddh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half{{.*}} [[B:%.*]]) #[[ATTR0:[0-9]+]] {
// LLVM:  [[ADD:%.*]] = fadd half [[A]], [[B]]
// LLVM:  ret half [[ADD]]
  return vaddh_f16(a, b);
}

// LLVM-LABEL: @test_vsubh_f16(
// CIR-LABEL: @test_vsubh_f16(
float16_t test_vsubh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.]], half {{.*}} [[B:%.]]) #[[ATTR0:[0-9]+]] {
// LLVM:  [[SUB:%.*]] = fsub half [[A]], [[B]]
// LLVM:  ret half [[SUB]]
  return vsubh_f16(a, b);
}

// LLVM-LABEL: @test_vmulh_f16(
// CIR-LABEL: @test_vmulh_f16(
float16_t test_vmulh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.]], half {{.*}} [[B:%.]]) #[[ATTR0:[0-9]+]] {
// LLVM:  [[MUL:%.*]] = fmul half [[A]], [[B]]
// LLVM:  ret half [[MUL]]
  return vmulh_f16(a, b);
}

// LLVM-LABEL: @test_vdivh_f16(
// CIR-LABEL: @test_vdivh_f16(
float16_t test_vdivh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.]], half {{.*}} [[B:%.]]) #[[ATTR0:[0-9]+]] {
// LLVM:  [[DIV:%.*]] = fdiv half [[A]], [[B]]
// LLVM:  ret half [[DIV]]
  return vdivh_f16(a, b);
}
