#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

AC_DEFUN([CHECK_DIETGPU],[

AS_IF([test "x$dietgpu_checked" != "xyes"],[
    dietgpu_happy="no"
    AC_ARG_WITH([dietgpu],
            [AS_HELP_STRING([--with-dietgpu=(DIR)], [Enable the use of DietGPU.])],
            [], [with_sharp=guess])

    AS_IF([test "x$with_dietgpu" != "xno"], [
        AC_SUBST(DIETGPU_CPPFLAGS, "-I$with_dietgpu/include/ ")
        AC_SUBST(DIETGPU_LDFLAGS, "-ldietgpu -L$with_dietgpu/lib")
        dietgpu_happy="yes"
        AC_DEFINE([HAVE_DIETGPU], 1, [Enable DietGPU support])],
        dietgpu_happy="yes"
    [
        AC_MSG_WARN([DietGPU is disabled])
        dietgpu_happy="no"
    ])
    AM_CONDITIONAL([HAVE_DIETGPU], [test "x$dietgpu_happy" != xno])
    dietgpu_checked=yes
])
])
