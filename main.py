import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import new_train
import glob
import os.path

class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
               'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
               'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Banana', 'Banana Lady Finger',
               'Banana Red', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red',
               'Cherry Wax Yellow', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3',
               'Grape White 4', 'Pear', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red',
               'Pear Williams']
class_dict = {}
fruit_dict = {}
for fruit in class_names:
    big_class = fruit.split()[0]
    fruit_dict[fruit] = big_class


def predict_image(file_name, model_path='train_models/best_model.pth'):
    Device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')

    predict_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(file_name).convert('RGB')

    model = new_train.create_model(fine_tune=True, num_classes=len(class_names), num_layer=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    image = predict_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()

    predicted_class = class_names[np.argmax(probabilities)]
    poss = probabilities[np.argmax(probabilities)]
    print(f'预测类别为：{fruit_dict[predicted_class]},可信度为：{poss:.4f}')


# 用于处理同一个文件夹下的所有图片
def predict_image_butch(folder_name, model_path='train_models/best_model.pth'):
    Device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')

    predict_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_paths = glob.glob(os.path.join(folder_name, "*[jp][pn]g"))
    batch_image = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        batch_image.append(predict_transform(image))
    batch_tensor = torch.stack(batch_image)

    model = new_train.create_model(fine_tune=True, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs, predicted_classes = torch.max(torch.softmax(outputs, dim=1), dim=1)

    for i in range(len(predicted_classes)):
        print(f"预测水果种类为：{fruit_dict[class_names[predicted_classes[i]]]}，具体为：{class_names[predicted_classes[i]]}，"
              f"可信度为：{probs[i]:.4f}；源图片：{os.path.basename(image_paths[i])}")


if __name__ == '__main__':
    predict_image(file_name='fruit_image/application_image/aaa1.jpg')

