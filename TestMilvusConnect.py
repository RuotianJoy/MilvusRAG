#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milvus连接测试脚本
用于测试与Milvus向量数据库的连接是否正常
"""

import os
import sys
from pymilvus import connections, utility
import time

# 连接参数
HOST = '47.115.47.33'
PORT = '19530'


def test_milvus_connection():
    """测试Milvus连接状态"""
    print(f"尝试连接Milvus服务器: {HOST}:{PORT}")

    try:
        # 尝试建立连接
        connections.connect(
            alias="default",
            host=HOST,
            port=PORT
        )

        # 验证连接
        if utility.has_collection("random_name_for_test"):
            print("连接成功: 能够查询集合信息")
        else:
            print("连接成功: 服务器正常响应")

        # 获取Milvus版本信息
        server_version = utility.get_server_version()
        print(f"Milvus服务器版本: {server_version}")

        # 列出所有集合
        collections = utility.list_collections()
        if collections:
            print(f"现有集合列表 ({len(collections)}个):")
            for i, collection in enumerate(collections):
                print(f"  {i + 1}. {collection}")
        else:
            print("当前没有创建任何集合")

        return True

    except Exception as e:
        print(f"连接失败: {str(e)}")

        # 提供一些可能的解决方案
        print("\n可能的解决方案:")
        print("1. 检查Milvus服务是否正在运行")
        print("2. 确认主机名和端口配置正确")
        print("3. 检查网络连接和防火墙设置")
        print("4. 如果使用Docker，确认容器正在运行并且端口已正确映射")
        print("   可以使用 'docker ps' 命令检查Milvus容器状态")

        return False

    finally:
        # 断开连接
        try:
            connections.disconnect("default")
            print("已断开与Milvus的连接")
        except:
            pass


if __name__ == "__main__":
    # 允许通过命令行参数传递主机和端口
    if len(sys.argv) > 1:
        HOST = sys.argv[1]
    if len(sys.argv) > 2:
        PORT = sys.argv[2]

    # 测试连接
    start_time = time.time()
    result = test_milvus_connection()
    end_time = time.time()

    # 输出测试结果和耗时
    print(f"\n测试耗时: {end_time - start_time:.2f}秒")
    print(f"连接测试{'成功' if result else '失败'}")

    # 返回适当的退出码
    sys.exit(0 if result else 1)