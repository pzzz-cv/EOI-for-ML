import torch
import numpy as np
def edge_screen(edge_flow, image_flow, kernel_size = 5):
    edge_flow = torch.from_numpy(np.float32(edge_flow)).cuda()
    image_flow = torch.from_numpy(np.float32(image_flow)).cuda()
    kernel_cus = np.ones([5,5])
    kernel_cus = torch.FloatTensor(kernel_cus).expand(1, 1, kernel_size,kernel_size)
    weight_cus = torch.nn.Parameter(data=kernel_cus, requires_grad=False).cuda()
    grow_map = torch.nn.functional.conv2d(edge_flow.unsqueeze(0).unsqueeze(0), weight_cus, stride=1, padding=int((kernel_size-1)/2), groups=1)
    grow_map = grow_map >= 1/(kernel_size*kernel_size)
    image_flow = image_flow == 1
    output = image_flow*grow_map.squeeze(0).squeeze(0)
    return output, grow_map.squeeze(0).squeeze(0)

if __name__ == '__main__':
    edge_flow = torch.ones([360,640]).cuda()
    image_flow = torch.ones([360,640]).cuda()
    print(edge_screen(edge_flow, image_flow, kernel_size = 5))