import json

# 修改为你实际的 JSON 路径
json_path = r"D:\你的路径\train.json" 

with open(json_path, 'r') as f:
    data = json.load(f)

max_id = -1
for ann in data['annotations']:
    if ann['category_id'] > max_id:
        max_id = ann['category_id']

print(f"数据集中最大的 category_id 是: {max_id}")
print(f"请确保你的配置文件中 num_classes > {max_id}")