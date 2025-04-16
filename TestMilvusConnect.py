#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milvus连接测试脚本
用于测试与Milvus向量数据库的连接是否正常
"""

import os
import sys
from pymilvus import connections, utility, Collection, DataType
import time
import json
from tabulate import tabulate

# 连接参数
HOST = '127.0.0.1'
PORT = '19530'

# 数据展示的配置
SAMPLE_ROWS = 5  # 每个集合展示的样本数据行数


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
                print(f"\n{'-'*50}")
                print(f"集合 {i + 1}: {collection}")
                print(f"{'-'*50}")
                
                # 获取并打印集合的字段和维度信息
                fields_info = print_collection_schema(collection)
                
                # 获取并展示集合数据
                if fields_info:
                    show_collection_data(collection, fields_info)
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


def print_collection_schema(collection_name):
    """打印集合的字段信息和向量维度"""
    try:
        # 获取集合对象
        collection = Collection(collection_name)
        
        # 获取集合的schema
        schema = collection.schema
        
        # 打印字段信息
        print(f"【字段结构】")
        fields = schema.fields
        fields_info = []
        table_data = []
        
        for field in fields:
            field_type = field.dtype
            field_name = field.name
            is_primary = field.is_primary
            
            field_info = {
                "name": field_name,
                "type": str(field_type),
                "is_primary": is_primary,
                "params": field.params,
                "is_vector": False
            }
            
            # 检查是否为向量字段，并获取维度
            if str(field_type).startswith('DataType.FLOAT_VECTOR') or str(field_type).startswith('DataType.BINARY_VECTOR'):
                dimension = field.params.get('dim')
                field_info["is_vector"] = True
                field_info["dimension"] = dimension
                table_row = [field_name, str(field_type), "是" if is_primary else "否", f"维度: {dimension}"]
            else:
                table_row = [field_name, str(field_type), "是" if is_primary else "否", "-"]
                
            fields_info.append(field_info)
            table_data.append(table_row)
        
        # 使用tabulate打印表格
        print(tabulate(table_data, headers=["字段名", "数据类型", "是否主键", "附加信息"], tablefmt="grid"))
        print("")  # 添加空行以提高可读性
        
        # 获取并打印集合的行数
        row_count = collection.num_entities
        print(f"集合行数: {row_count:,}")
        
        return fields_info
        
    except Exception as e:
        print(f"无法获取集合 '{collection_name}' 的schema: {str(e)}")
        return None


def show_collection_data(collection_name, fields_info):
    """展示集合中的数据样本"""
    try:
        # 获取集合对象
        collection = Collection(collection_name)
        
        # 准备查询条件
        non_vector_fields = [field["name"] for field in fields_info if not field["is_vector"]]
        
        if not non_vector_fields:
            print("该集合没有非向量字段，无法展示数据")
            return
        
        # 获取数据样本
        print(f"\n【数据样本】(最多 {SAMPLE_ROWS} 行)")
        collection.load()
        
        try:
            results = collection.query(
                expr="",
                output_fields=non_vector_fields,
                limit=SAMPLE_ROWS
            )
            
            if not results:
                print("没有查询到数据")
                return
            
            # 准备表格数据
            table_data = []
            for item in results:
                row = []
                for field in non_vector_fields:
                    value = item.get(field, "")
                    
                    # 处理长文本
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    elif isinstance(value, (list, dict)):
                        value = str(value)[:47] + "..." if len(str(value)) > 50 else str(value)
                        
                    row.append(value)
                    
                table_data.append(row)
            
            # 使用tabulate打印表格
            print(tabulate(table_data, headers=non_vector_fields, tablefmt="grid"))
            
        finally:
            collection.release()
        
    except Exception as e:
        print(f"无法获取集合 '{collection_name}' 的数据样本: {str(e)}")


if __name__ == "__main__":
    # 允许通过命令行参数传递主机和端口
    if len(sys.argv) > 1:
        HOST = sys.argv[1]
    if len(sys.argv) > 2:
        PORT = sys.argv[2]
    
    # 尝试导入tabulate，如果不存在则提示安装
    try:
        import tabulate
    except ImportError:
        print("缺少依赖库: tabulate")
        print("请使用以下命令安装: pip install tabulate")
        sys.exit(1)

    # 测试连接
    start_time = time.time()
    result = test_milvus_connection()
    end_time = time.time()

    # 输出测试结果和耗时
    print(f"\n{'-'*50}")
    print(f"测试耗时: {end_time - start_time:.2f}秒")
    print(f"连接测试{'成功' if result else '失败'}")

    # 返回适当的退出码
    sys.exit(0 if result else 1)