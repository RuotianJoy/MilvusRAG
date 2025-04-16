#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import sys
import os

# 中英文学校名称映射表
school_mapping = {
    # 英国大学
    "剑桥大学": "University of Cambridge",
    "牛津大学": "University of Oxford",
    "帝国理工学院": "Imperial College London",
    "伦敦大学学院": "University College London",
    "爱丁堡大学": "University of Edinburgh",
    "曼彻斯特大学": "University of Manchester",
    "伦敦国王学院": "King's College London",
    "伦敦政治经济学院": "London School of Economics and Political Science",
    "华威大学": "University of Warwick",
    "布里斯托大学": "University of Bristol",
    
    # 美国大学
    "哈佛大学": "Harvard University",
    "斯坦福大学": "Stanford University",
    "麻省理工学院": "Massachusetts Institute of Technology",
    "加州理工学院": "California Institute of Technology",
    "普林斯顿大学": "Princeton University",
    "芝加哥大学": "University of Chicago",
    "耶鲁大学": "Yale University",
    "康奈尔大学": "Cornell University",
    "宾夕法尼亚大学": "University of Pennsylvania",
    "加州大学伯克利分校": "University of California, Berkeley",
    "加州大学洛杉矶分校": "University of California, Los Angeles",
    "约翰霍普金斯大学": "Johns Hopkins University",
    "密歇根大学安娜堡分校": "University of Michigan-Ann Arbor",
    "哥伦比亚大学": "Columbia University",
    "杜克大学": "Duke University",
    "纽约大学": "New York University",
    "西北大学": "Northwestern University",
    "卡内基梅隆大学": "Carnegie Mellon University",
    
    # 加拿大大学
    "多伦多大学": "University of Toronto",
    "不列颠哥伦比亚大学": "University of British Columbia",
    "麦吉尔大学": "McGill University",
    "阿尔伯塔大学": "University of Alberta",
    "蒙特利尔大学": "University of Montreal",
    
    # 亚洲大学
    "清华大学": "Tsinghua University",
    "北京大学": "Peking University",
    "复旦大学": "Fudan University",
    "浙江大学": "Zhejiang University",
    "上海交通大学": "Shanghai Jiao Tong University",
    "南京大学": "Nanjing University",
    "中国科学技术大学": "University of Science and Technology of China",
    "香港大学": "University of Hong Kong",
    "香港中文大学": "Chinese University of Hong Kong",
    "香港科技大学": "Hong Kong University of Science and Technology",
    "新加坡国立大学": "National University of Singapore",
    "东京大学": "University of Tokyo",
    "京都大学": "Kyoto University",
    "大阪大学": "Osaka University",
    "首尔国立大学": "Seoul National University",
    
    # 澳大利亚大学
    "墨尔本大学": "University of Melbourne",
    "悉尼大学": "University of Sydney",
    "澳大利亚国立大学": "Australian National University",
    "昆士兰大学": "University of Queensland",
    "新南威尔士大学": "University of New South Wales",
    "莫纳什大学": "Monash University",
    
    # 欧洲大学
    "苏黎世联邦理工学院": "ETH Zurich",
    "巴黎萨克雷大学": "Paris-Saclay University",
    "巴黎第六大学": "Sorbonne University",
    "慕尼黑工业大学": "Technical University of Munich",
    "海德堡大学": "Heidelberg University",
    "鹿特丹伊拉斯姆斯大学": "Erasmus University Rotterdam",
    "洛桑联邦理工学院": "EPFL",
    "巴黎高等师范学院": "École Normale Supérieure - Paris",
    "巴黎高等理工学院": "École Polytechnique"
}

def translate_school_names(text):
    """将文本中的中文学校名称替换为英文名称"""
    if not isinstance(text, str):
        return text
        
    # 对映射表中的每个学校名称进行替换
    for cn_name, en_name in school_mapping.items():
        text = text.replace(cn_name, en_name)
        
    return text

def process_excel_file(input_file, output_file=None):
    """处理Excel文件，将第二列中与THE2025有关的中文学校名称改为英文"""
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)
        
        # 获取列名列表
        columns = df.columns.tolist()
        
        if len(columns) < 2:
            print(f"错误: Excel文件'{input_file}'至少需要2列")
            return False
            
        # 假设第二列是Questions列
        second_column = columns[1]
        
        # 筛选出第二列中含有"THE2025"的行
        the_mask = df[second_column].astype(str).str.contains("THE", na=False)
        
        # 对这些行的第二列应用翻译函数
        df.loc[the_mask, second_column] = df.loc[the_mask, second_column].apply(translate_school_names)
        
        # 如果没有指定输出文件，则创建一个新文件名
        if output_file is None:
            base_name, ext = os.path.splitext(input_file)
            output_file = f"{base_name}_en{ext}"
        
        # 保存修改后的Excel文件
        df.to_excel(output_file, index=False)
        
        print(f"成功处理文件: {input_file}")
        print(f"已将结果保存至: {output_file}")
        print(f"已替换 {the_mask.sum()} 行中的学校名称")
        
        return True
        
    except Exception as e:
        print(f"处理文件'{input_file}'时出错: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python school_name_translator.py <input_excel_file> [output_excel_file]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"错误: 文件'{input_file}'不存在")
        sys.exit(1)
        
    success = process_excel_file(input_file, output_file)
    sys.exit(0 if success else 1) 