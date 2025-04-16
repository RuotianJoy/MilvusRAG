# Update the answers by translating Chinese university names into English
data['Answers'] = data['Answers'].replace({
    '哈佛大学': 'Harvard University',
    '斯坦福大学': 'Stanford University',
    '麻省理工学院': 'Massachusetts Institute of Technology (MIT)',
    '剑桥大学': 'University of Cambridge',
    '加州大学': 'University of California',
    '新加坡国立大学': 'National University of Singapore (NUS)',
    '清华大学': 'Tsinghua University',
    '北京大学': 'Peking University',
    '复旦大学': 'Fudan University',
    '牛津大学': 'University of Oxford',
    '巴黎萨克雷大学': 'Paris-Saclay University',
    '巴黎第六大学': 'University of Paris VI'
}, regex=True)

# Display the updated data
import ace_tools as tools; tools.display_dataframe_to_user(name="Updated THE and ARWU Data", dataframe=data)
