#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import configparser
from pymilvus import utility, connections

class MilvusCollectionDeleter:
    def __init__(self):
        """初始化Milvus集合删除器"""
        # 获取项目根目录
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 连接到Milvus服务
        self.connect_to_milvus()
        
    def load_config(self):
        """读取配置文件"""
        # 配置文件路径
        config_file = os.path.join(self.project_root, "Config", "Milvus.ini")
        if not os.path.exists(config_file):
            print(f"错误：配置文件不存在: {config_file}")
            return {
                'host': 'localhost',
                'port': '19530'
            }
            
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        return {
            'host': config.get('connection', 'host', fallback='localhost'),
            'port': config.get('connection', 'port', fallback='19530')
        }
    
    def connect_to_milvus(self):
        """连接到Milvus服务器"""
        # 加载配置
        milvus_config = self.load_config()
        host = milvus_config['host']
        port = milvus_config['port']
        
        try:
            connections.connect("default", host=host, port=port)
            print(f"已成功连接到Milvus服务器: {host}:{port}")
        except Exception as e:
            print(f"连接Milvus服务器失败: {str(e)}")
            sys.exit(1)
    
    def list_collections(self):
        """列出所有集合"""
        try:
            collections = utility.list_collections()
            if not collections:
                print("Milvus中没有集合")
                return []
            
            print("Milvus中的集合列表:")
            for i, coll in enumerate(collections, 1):
                print(f"{i}. {coll}")
            
            return collections
        except Exception as e:
            print(f"获取集合列表失败: {str(e)}")
            return []
    
    def delete_collection(self, collection_name):
        """删除指定名称的集合"""
        if not collection_name:
            print("错误：未指定集合名称")
            return False
        
        try:
            # 检查集合是否存在
            if not utility.has_collection(collection_name):
                print(f"集合 '{collection_name}' 不存在")
                return False
            
            # 删除集合
            utility.drop_collection(collection_name)
            print(f"已成功删除集合 '{collection_name}'")
            return True
        except Exception as e:
            print(f"删除集合 '{collection_name}' 失败: {str(e)}")
            return False
    
    def delete_multiple_collections(self, collection_names):
        """删除多个集合"""
        if not collection_names:
            print("错误：未指定任何集合名称")
            return False
        
        success_count = 0
        for name in collection_names:
            if self.delete_collection(name):
                success_count += 1
        
        print(f"操作完成，成功删除 {success_count}/{len(collection_names)} 个集合")
        return success_count == len(collection_names)

def main():
    parser = argparse.ArgumentParser(description="删除Milvus集合工具")
    parser.add_argument("--list", action="store_true", help="列出所有集合")
    parser.add_argument("--delete", action="store", help="要删除的集合名称")
    parser.add_argument("--delete-all", action="store_true", help="删除所有集合")
    parser.add_argument("--prefix", action="store", help="删除指定前缀的所有集合")
    args = parser.parse_args()
    
    # 创建删除器实例
    deleter = MilvusCollectionDeleter()
    
    # 如果只是列出集合
    if args.list:
        deleter.list_collections()
        return
    
    # 删除单个集合
    if args.delete:
        deleter.delete_collection(args.delete)
        return
    
    # 删除所有集合
    if args.delete_all:
        collections = deleter.list_collections()
        if collections:
            confirm = input("确认要删除所有集合吗? (y/n): ")
            if confirm.lower() == 'y':
                deleter.delete_multiple_collections(collections)
            else:
                print("操作已取消")
        return
    
    # 删除指定前缀的集合
    if args.prefix:
        collections = deleter.list_collections()
        if collections:
            # 过滤出符合前缀的集合
            prefix_collections = [c for c in collections if c.startswith(args.prefix)]
            
            if not prefix_collections:
                print(f"没有找到前缀为 '{args.prefix}' 的集合")
                return
            
            print(f"将删除以下集合:")
            for c in prefix_collections:
                print(f"- {c}")
            
            confirm = input(f"确认要删除这 {len(prefix_collections)} 个集合吗? (y/n): ")
            if confirm.lower() == 'y':
                deleter.delete_multiple_collections(prefix_collections)
            else:
                print("操作已取消")
        return
    
    # 如果没有提供任何参数，则显示帮助
    if not (args.list or args.delete or args.delete_all or args.prefix):
        parser.print_help()

if __name__ == "__main__":
    main() 