import torch
import torch.quantization as quantization
import network

# Load your pre-trained model
with torch.no_grad():

        model = network.model_dict['FCN4_Deep_Resize_Enc'](upsample_mode='nearest')
        sd = torch.load('/media/pi/YI CARCAM/gamma01.pth', map_location=torch.device('cpu'))['model']
        ud = torch.load('/media/pi/YI CARCAM/gamma01.pth', map_location=torch.device('cpu'))
        # print(sd.keys())
        encoder = ud['encoder']
        
        model.load_state_dict(sd)

        # for name, module in model.named_modules():
        # # 'name' is the name or path of the module, 'module' is the actual module object
        #     print(f"Module name: {name}")
        #     print(f"Module type: {type(module)}")
        #     print(module)

        # Define quantization configurations for different layer types
        qconfig_conv = quantization.get_default_qconfig('fbgemm')
        qconfig_linear = quantization.get_default_qconfig('qnnpack')

        # Create a dictionary to map layer types to their quantization configuration
        qconfig_dict = {
            torch.nn.Conv2d: qconfig_conv,
            torch.nn.Linear: qconfig_linear,
        }

        quantized_model = quantization.quantize_dynamic(
            model, qconfig_spec=qconfig_dict, dtype=torch.qint8
        )
        quantized_model.eval()
        quant_model = {'model':quantized_model.state_dict(),
                       'encoder': encoder}
        
        # Save the quantized model
        torch.save(quant_model, '/home/pi/Desktop/USCT/models/TaskBased/quant_gamma01.pth')


# import torch
# from torch import nn
# import torch.nn.utils.prune as prune

# # Load your pre-trained model
# model = torch.load('/home/pi/Desktop/USCT/models/TaskBased/task.pth', map_location=torch.device('cpu'))
# print(model)

# # Specify the pruning method and parameters
# pruning_method = prune.L1Unstructured
# parameters_to_prune = (
#     (model.conv1, 'weight'),
#     (model.fc1, 'weight'),
# )

# # Apply pruning to the model
# for layer, param_name in parameters_to_prune:
#     prune.global_unstructured(
#         layer, pruning_method, name=param_name, amount=0.2
#     )

# # Remove the pruning re-parametrization
# for layer, param_name in parameters_to_prune:
#     prune.remove(layer, param_name)

# # Save the pruned model
# torch.save(model, '/home/pi/Desktop/USCT/models/TaskBased/quant_task.pth')