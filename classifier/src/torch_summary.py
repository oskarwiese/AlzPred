from torchsummary import summary
import classifier_model

# torch.Size([1, 1, 256, 145, 121])


model = classifier_model.ConvNet()
summary(model, (1, 256, 145, 121))