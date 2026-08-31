# 面片拓扑分类及分析方法

## 1. 概述

`src/toys3d/geometrics.py` 提供了一套面片（face）级别的拓扑缺陷检测、分类、邻域统计与水密修复工具。`src/toys3d/meshfilter.py` 在这些拓扑关系之上实现了 k-ring 邻域滤波。

典型诊断流程：

