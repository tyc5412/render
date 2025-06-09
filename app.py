import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import gradio as gr

# 模型加载（可替换为你的模型）
model = resnet18(pretrained=True)
model.eval()

# 输入图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 示例标签（你可以替换为自己的）
labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]

# 推理函数
def classify(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(img)
        probs = torch.nn.functional.softmax(preds[0], dim=0)
    top = torch.topk(probs, 3)
    return {labels[i]: float(top.values[j]) for j, i in enumerate(top.indices)}

# 启动 Gradio 界面
iface = gr.Interface(fn=classify, inputs=gr.Image(type="pil"), outputs="label", title="图像分类Demo")
iface.launch(server_name="0.0.0.0", server_port=8080)
