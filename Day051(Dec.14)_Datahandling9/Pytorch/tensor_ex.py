import torch
import numpy as np

# data = [[1,2], [3,4]]
# x_data = torch.tensor(data)
# # print(x_data)
# # print(x_data.shape)



# # numpy 배열로부터 생성
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
# # print(x_np)

# x_ones  = torch.ones_like(x_data)
# # print(f"Ones Tensor >> \n", x_ones)

# x_rand = torch.rand_like(x_data, dtype=torch.float)
# # print(x_rand)
# # torch.randn 도 있음
# # torch.manual_seed()

# shape = (4,6)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# # print(f"Random Tensor: \n", rand_tensor)
# # print(f"Ones Tensor: \n", ones_tensor)
# # print(f"Zeros Tensor: \n", zeros_tensor)

# tensor = torch.rand(4,5)
# # print(f"Shape of tensor: {tensor.shape}")
# # print(f"Datatype of tensor: {tensor.dtype}")
# # print(f"Device tensor is stored on: {tensor.device}")

# # Shape of tensor: torch.Size([4, 5])
# # Datatype of tensor: torch.float32
# # Device tensor is stored on: cpu



# # GPU 존재시 텐서 이동방법
# tensor = torch.rand(3,4)
# if torch.cuda.is_available():
#     tensor = tensor.to('cuda')
# else:
#     tensor = tensor.to("cpu")

# # print(f"Device tensor is stored on: {tensor.device}")
# # Device tensor is stored on: cpu

# tensor = torch.ones(5,5)
# # tensor[:,3] = 0
# # tensor[:,:3] = 0
# tensor[3,:] = 0
# print(tensor)



# # 텐서 합치기
# tensor = torch.ones(5,5)
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# # print(t1)

# # 텐서 곱
# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# print(tensor * tensor)

# # 두 텐서간 행렬 곱
# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# # 다른 문법:
# print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# # 바꿔치기(in-place)
# print(tensor, "\n")
# tensor.add_(5)  # 접미사가 있어야만 바꿔치기 연산이 가능
# print(tensor)



# # NumPy 변환(Bridge)
# # 텐서를 Numpy 배열로 변환
# t = torch.ones(5)
# print(t)    # tensor([1., 1., 1., 1., 1.])
# n = t.numpy()
# print(n)    # [1. 1. 1. 1. 1.]

# # 텐서의 변경 사항이 Numpy배열에 반영
# t.add_(6)
# print(t)    # tensor([7., 7., 7., 7., 7.])
# print(n)    # [7. 7. 7. 7. 7.]

# # Numpy 배열을 텐서로 변환
# n = np.ones(5)  
# t = torch.from_numpy(n)
# print(n)    # [1. 1. 1. 1. 1.]
# print(t)    # tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

# np.add(n, 6, out=n)
# print(n)    # [7. 7. 7. 7. 7.]
# print(t)    # tensor([7., 7., 7., 7., 7.], dtype=torch.float64)



# # 뷰(View) 원소의 수를 유지하면서 텐서의 크기만 변경. 매우 중요
# # 넘파이 Reshape과 같은 역할
# t = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])
# ft = torch.FloatTensor(t)
# # print(ft.shape)
# # print(ft)

# # print(ft.view([-1,3]))      # ft라는 텐서를 (?, 3)의 크기로 변경
# # print(ft.view([-1,3]).shape)    # torch.Size([4, 3])
# # 내부적 크기 변환 (2, 2, 3) -> (2 × 2, 3) -> (4, 3)
# """
# • view는 기본적으로 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 합니다.
# • 파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추합니다.
# 변경 전 텐서의 원소의 수는 (2 × 2 × 3) = 12개였습니다. 그리고 변경 후 텐서의 원소의 개수 또한 (4 × 3) = 12개였
# 습니다.
# """
# """
# 3차원 텐서의 크기 변경
# 이번에는 3차원 텐서에서 3차원 텐서로 차원은 유지하되, 크기(shape)를 바꾸는 작업을 해보겠습니다. view로 텐
# 서의 크기를 변경하더라도 원소의 수는 유지되어야 한다고 언급한 바 있습니다. 
# 그렇다면 (2 × 2 × 3) 텐서를 (? × 1 × 3) 텐서로 변경하라고 하면 ?는 몇 차원인가요?
# (2 × 2 × 3) = (? × 1 × 3) = 12를 만족해야 하므로 ?는 4가 됩니다. 이를 실습으로 확인해봅시다.
# """
# # print(ft.view([-1,1,3]))
# # print(ft.view([-1,1,3]).shape)  # torch.Size([4, 1, 3])



# # 스퀴즈(Squeeze) 1인 차원 제거
# # 스퀴즈는 차원이 1인 경우에는 해당 차원을 제거합니다.
# # 실습을 위해 임의로 (3 × 1)의 크기를 가지는 2차원 텐서를 만들겠습니다.

# ft= torch.FloatTensor([[0],[1],[2]])
# # print(ft)
# # print(ft.shape) # torch.Size([3, 1])

# # 해당 텐서는 (3 × 1)의 크기를 가집니다. 두번째 차원이 1이므로 squeeze를 사용하면 (3,)의 크기를 가지는
# # 텐서로 변경됩니다.

# # print(ft.squeeze())
# # print(ft.squeeze().shape)   # torch.Size([3])

# 언스퀴즈(Unsqueeze) 특정 위치에 1인 차원 추가
ft = torch.Tensor([0,1,2])
# print(ft)
# print(ft.shape)
"""
현재는 차원이 1개인 1차원 벡터입니다. 여기에 첫번째 차원에 1인 차원을 추가해보겠습니다. 첫번째 차원의 인
덱스를 의미하는 숫자 0을 인자로 넣으면 첫번째 차원에 1인 차원이 추가됩니다
"""
# print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
# print(ft.unsqueeze(0).shape)    # torch.Size([1, 3])
# print(ft.unsqueeze(1).shape)    # torch.Size([3, 1])
# print(ft.unsqueeze(-1).shape)   # torch.Size([3, 1])

# print(ft.view(1,-1))
# print(ft.view(1,-1).shape)  # torch.Size([1, 3])

